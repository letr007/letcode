use crate::agent::ConversationMessage;
use crate::agent::{
    CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION, ContextCompactionDerivedCoverage,
    ContextCompactionDerivedCoverageItem, ContextCompactionDerivedKind,
    ContextCompactionFrameBinding, ContextCompactionSourceSpan,
};
use crate::context_view::{self, ContextViewProjection};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::{ProtocolFrame, history_items_to_frames};
use crate::request_builder::HistoryItem;
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameId, RuntimeFrameKind, RuntimeSnapshot,
    RuntimeSource, SourceSpan,
};
use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use anyhow::{anyhow, ensure};
use std::collections::{BTreeMap, BTreeSet};

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
    checkpoint_spans_from_history, checkpoint_spans_to_compaction, merge_source_spans,
    restore_history_projection,
};

#[path = "transcript_projection/checkpoint.rs"]
mod checkpoint;

pub(crate) use checkpoint::{
    prepare_logical_checkpoint_candidate, validate_logical_checkpoint_candidate,
};

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
/// A checkout establishes a durable cursor scope even when its branch later
/// advances. Sequence zero keeps old transcripts deterministic.
fn context_scope_revision(_records: &[TranscriptRecord], resolved: &ResolvedBranchContext) -> u64 {
    resolved.scope_checkout_sequence.unwrap_or(0)
}
/// Metadata is append-only and normally has no branch id. Its ownership is
/// therefore bounded by the checkout that selected the active scope and the
/// next checkout that replaces it.
/// Reconstruct the pre-compaction protocol frames from the append-only journal.
/// The active history projection intentionally replaces these frames with a
/// summary; RuntimeSnapshot retains them as retired identity/provenance records.
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
            TranscriptEvent::ContextCompaction(event) if event.outcome == "succeeded" => {
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

/// Resolves persisted retirement spans, preserving the legacy fallback for
/// compaction records written before spans became durable.
pub(super) fn resolved_compaction_retired_source_spans(
    prior_records: &[TranscriptRecord],
    event: &crate::agent::ContextCompactionEvent,
) -> Vec<ContextCompactionSourceSpan> {
    if event.retired_source_spans.is_empty() {
        derive_retired_source_spans(prior_records, event.tail_start_index)
    } else {
        event.retired_source_spans.clone()
    }
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

pub(crate) const RETAINED_FACTS_MARKER: &str = "[retained-facts:v1]";
pub(crate) const RETAINED_FACTS_DELIMITER: &str = "\n\n[retained-facts:v1]\n";
pub(crate) const MAX_RETAINED_FACT_TEXT_BYTES: usize = 4 * 1024;
pub(crate) const MAX_RETAINED_FACTS_BYTES: usize = MAX_RETAINED_FACT_TEXT_BYTES * 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactionClosure {
    pub co_retired_frame_ids: BTreeSet<RuntimeFrameId>,
    pub coverage_frame_ids: BTreeSet<RuntimeFrameId>,
}

/// The sole projection closure classifier.  Co-retirement means a frame has no
/// remaining raw-source authority; typed coverage is deliberately narrower.
pub(crate) fn classify_compaction_closure(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
) -> CompactionClosure {
    classify_compaction_closure_with_mode(
        snapshot,
        retired_spans,
        CompactionClosureMode::Standard,
    )
}

/// Controls whether contributor-derived protected projection frames may be
/// co-retired when their source spans are fully covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionClosureMode {
    /// Keep raw-source-retaining projections (and their sources) intact.
    Standard,
    /// Under request pressure, fully covered non-protocol projections may be
    /// co-retired even if a retaining contributor listed them in
    /// `protected_frame_ids`. Turn/explicit protection still blocks retirement.
    RequestPressure,
}

pub(crate) fn classify_compaction_closure_with_mode(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
    mode: CompactionClosureMode,
) -> CompactionClosure {
    // Contributor-derived ids join `protected_frame_ids` for request identity.
    // Under request pressure they must not permanently block co-retirement of
    // fully covered projections, or selection reports
    // `protected_context,retained_source_dependency` with no retirable prefix.
    let protected = match mode {
        CompactionClosureMode::Standard => snapshot
            .compaction
            .protected_frame_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        CompactionClosureMode::RequestPressure => snapshot
            .compaction
            .explicit_protected_frame_ids
            .iter()
            .copied()
            .chain(
                snapshot
                    .compaction
                    .turn_protected_frame_ids
                    .iter()
                    .copied(),
            )
            .collect::<BTreeSet<_>>(),
    };
    let coverage = snapshot
        .prompt_contributors
        .iter()
        .filter(|contributor| !contributor.retains_raw_sources)
        .flat_map(|contributor| contributor.frame_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let co_retired_frame_ids = snapshot
        .frames
        .iter()
        .filter(|frame| {
            matches!(
                frame.visibility,
                FrameVisibility::Active | FrameVisibility::Folded
            ) && frame.protocol.is_none()
                && frame.provenance.source != RuntimeSource::SummaryArtifact
                && frame
                    .provenance
                    .source_span
                    .is_some_and(|span| span.covered_by_any(retired_spans))
                && !protected.contains(&frame.id)
        })
        .map(|frame| frame.id)
        .collect::<BTreeSet<_>>();
    CompactionClosure {
        coverage_frame_ids: co_retired_frame_ids
            .intersection(&coverage)
            .copied()
            .collect(),
        co_retired_frame_ids,
    }
}

pub(crate) fn derive_modern_compaction_coverage(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
) -> anyhow::Result<(CompactionClosure, ContextCompactionDerivedCoverage)> {
    derive_modern_compaction_coverage_with_mode(
        snapshot,
        retired_spans,
        CompactionClosureMode::Standard,
    )
}

pub(crate) fn derive_modern_compaction_coverage_with_mode(
    snapshot: &RuntimeSnapshot,
    retired_spans: &[SourceSpan],
    mode: CompactionClosureMode,
) -> anyhow::Result<(CompactionClosure, ContextCompactionDerivedCoverage)> {
    let closure = classify_compaction_closure_with_mode(snapshot, retired_spans, mode);
    let mut items = Vec::new();
    for frame in snapshot
        .frames
        .iter()
        .filter(|frame| closure.coverage_frame_ids.contains(&frame.id))
    {
        let span = frame
            .provenance
            .source_span
            .ok_or_else(|| anyhow!("covered projection lacks a source span"))?;
        let (kind, identity, retained_text) = match (frame.kind, frame.provenance.source) {
            (RuntimeFrameKind::ContextBlock, RuntimeSource::ContextView) => {
                let id = frame
                    .provenance
                    .source_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("derived context block lacks durable identity"))?;
                let block_id = context_view::ContextBlockId::new(id.to_owned())?;
                let block = snapshot
                    .context_view
                    .blocks
                    .get(&block_id)
                    .ok_or_else(|| anyhow!("derived context block '{id}' is missing"))?;
                let kind = match block.kind {
                    context_view::ContextBlockKind::CurrentUserRequirement => {
                        ContextCompactionDerivedKind::CurrentUserRequirement
                    }
                    context_view::ContextBlockKind::FileWriteFact => {
                        ContextCompactionDerivedKind::FileWriteFact
                    }
                    context_view::ContextBlockKind::TestResult => {
                        ContextCompactionDerivedKind::TestResult
                    }
                    _ => return Err(anyhow!("unsupported non-retaining context block '{id}'")),
                };
                (
                    kind,
                    format!("context-block:{id}"),
                    format!("{}: {}", block.title, block.detail),
                )
            }
            (RuntimeFrameKind::Metadata, RuntimeSource::Transcript) => {
                let id = frame
                    .provenance
                    .source_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("evidence lacks durable identity"))?;
                let evidence = snapshot
                    .evidence
                    .iter()
                    .find(|record| record.id == id)
                    .ok_or_else(|| anyhow!("evidence '{id}' is missing"))?;
                ensure!(
                    evidence.sequence == span.start_sequence
                        && span.start_sequence == span.end_sequence,
                    "evidence source span does not match its typed record"
                );
                (
                    ContextCompactionDerivedKind::Evidence,
                    format!("evidence:{id}"),
                    evidence.summary.clone(),
                )
            }
            (RuntimeFrameKind::FoldedOutput, RuntimeSource::FoldedOutput) => {
                let id = frame
                    .provenance
                    .source_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("folded output lacks durable identity"))?;
                let output = snapshot
                    .context_view
                    .folded_outputs
                    .get(id)
                    .ok_or_else(|| anyhow!("folded output '{id}' is missing"))?;
                let output_span = match (output.source_start_sequence, output.source_end_sequence) {
                    (Some(start), Some(end)) => SourceSpan::new(start, end)?,
                    (Some(start), None) => SourceSpan::new(start, start)?,
                    _ => return Err(anyhow!("folded output '{id}' lacks a source span")),
                };
                ensure!(
                    output_span == span,
                    "folded output '{id}' source span does not match its runtime frame"
                );
                let text = deterministic_folded_output_retained_text(output)?;
                (
                    ContextCompactionDerivedKind::FoldedOutput,
                    format!("folded-output:{id}"),
                    text,
                )
            }
            _ => return Err(anyhow!("unsupported non-retaining covered projection")),
        };
        validate_retained_fact_text(&retained_text)?;
        items.push(ContextCompactionDerivedCoverageItem {
            kind,
            identity,
            source_span: ContextCompactionSourceSpan {
                start_sequence: span.start_sequence,
                end_sequence: span.end_sequence,
            },
            retained_text,
        });
    }
    items.sort_by(|left, right| left.identity.cmp(&right.identity));
    Ok((
        closure,
        ContextCompactionDerivedCoverage {
            version: CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION,
            items,
        },
    ))
}

fn deterministic_folded_output_retained_text(
    output: &context_view::FoldedOutputMetadata,
) -> anyhow::Result<String> {
    ensure!(
        output.provider_fold_eligible,
        "folded output '{}' is not a complete raw provider artifact",
        output.output_id
    );
    ensure!(
        is_trusted_raw_folded_output_kind(&output.output_kind),
        "folded output '{}' has unsupported raw output kind '{}'",
        output.output_id,
        output.output_kind
    );
    Ok(bounded_raw_folded_output_excerpt(&output.content))
}

fn is_trusted_raw_folded_output_kind(kind: &str) -> bool {
    matches!(
        kind,
        // `text` is the legacy representation emitted for generic tool results.
        "text"
            | "tool_result"
            | "shell_output"
            | "tool_output"
            | "file_content"
            | "search_matches"
            | "mcp_text"
    )
}

fn bounded_raw_folded_output_excerpt(content: &str) -> String {
    const HEADER: &str = "[bounded raw folded-output excerpt; not a semantic summary]\n";
    let mut end = (MAX_RETAINED_FACT_TEXT_BYTES - HEADER.len()).min(content.len());
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    format!("{HEADER}{}", &content[..end])
}

fn validate_retained_fact_text(text: &str) -> anyhow::Result<()> {
    ensure!(
        !text.trim().is_empty() && text.len() <= MAX_RETAINED_FACT_TEXT_BYTES,
        "derived coverage text is empty or exceeds its bound"
    );
    ensure!(
        !text.contains(RETAINED_FACTS_MARKER),
        "derived coverage text contains reserved retained-facts marker"
    );
    Ok(())
}

/// Formats legacy summary-embedded coverage for fixture builders and replay
/// compatibility. Production compaction persists coverage only in the typed
/// event field; new callers must not use this helper.
pub(crate) fn append_retained_facts(
    summary: String,
    coverage: &ContextCompactionDerivedCoverage,
) -> anyhow::Result<String> {
    ensure!(
        !summary.contains(RETAINED_FACTS_MARKER),
        "context compaction summary contains reserved retained-facts marker"
    );
    for item in &coverage.items {
        validate_retained_fact_text(&item.retained_text)?;
    }
    let facts = serde_json::to_string(&coverage.items)?;
    ensure!(
        facts.len() <= MAX_RETAINED_FACTS_BYTES,
        "retained facts section exceeds its bound"
    );
    Ok(format!("{summary}{RETAINED_FACTS_DELIMITER}{facts}"))
}

/// Builds the provider-visible summary from the canonical event fields.
/// Modern events persist retained facts only in `derived_coverage`; historical
/// events already carry the legacy suffix and pass through unchanged.
pub(crate) fn project_compaction_summary(
    summary: &str,
    coverage: Option<&ContextCompactionDerivedCoverage>,
) -> anyhow::Result<String> {
    if summary.contains(RETAINED_FACTS_MARKER) {
        return Ok(summary.to_string());
    }
    match coverage {
        Some(coverage) if !coverage.items.is_empty() => {
            append_retained_facts(summary.to_string(), coverage)
        }
        _ => Ok(summary.to_string()),
    }
}

fn validate_retained_facts_suffix(
    summary: &str,
    coverage: &ContextCompactionDerivedCoverage,
) -> anyhow::Result<()> {
    for item in &coverage.items {
        validate_retained_fact_text(&item.retained_text)?;
    }
    ensure!(
        summary.matches(RETAINED_FACTS_DELIMITER).count() == 1,
        "context compaction summary must contain exactly one retained-facts delimiter"
    );
    ensure!(
        summary.matches(RETAINED_FACTS_MARKER).count() == 1,
        "context compaction summary must contain exactly one retained-facts marker"
    );
    let (_, suffix) = summary
        .split_once(RETAINED_FACTS_DELIMITER)
        .expect("checked delimiter");
    ensure!(
        suffix.len() <= MAX_RETAINED_FACTS_BYTES,
        "retained facts section exceeds its bound"
    );
    let parsed: Vec<ContextCompactionDerivedCoverageItem> = serde_json::from_str(suffix)?;
    ensure!(
        serde_json::to_string(&parsed)? == suffix,
        "retained facts section must be canonical JSON"
    );
    ensure!(
        parsed == coverage.items,
        "retained facts section does not match typed coverage"
    );
    Ok(())
}

pub(crate) fn restore_retired_source_spans_projection(
    records: &[TranscriptRecord],
) -> Vec<ContextCompactionSourceSpan> {
    let mut retired = Vec::new();
    for record in records {
        if let TranscriptEvent::ContextCompaction(event) = &record.event
            && event.outcome == "succeeded"
        {
            let prior_records = &records[..records
                .iter()
                .position(|candidate| candidate.sequence == record.sequence)
                .unwrap_or(records.len())];
            retired.extend(resolved_compaction_retired_source_spans(prior_records, event));
        }
        if let TranscriptEvent::LogicalCheckpoint(event) = &record.event {
            retired.extend(checkpoint_spans_to_compaction(&event.covered_source_spans));
        }
    }
    canonical_cumulative_retired_source_spans(Vec::new(), retired)
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
    let records = scope.selected_history_records();
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
            newly_retired_source_spans.clone(),
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
    // Historical span-less replacement records predate typed coverage. Their
    // legacy inference remains readable; modern records never get this escape.
    if event.derived_coverage.is_none() && legacy_event {
        ensure!(
            !event.summary.contains(RETAINED_FACTS_MARKER),
            "legacy context compaction summary must not contain retained facts without coverage"
        );
        return Ok(());
    }
    let snapshot = runtime_snapshot_from_resolved_context_unbound(
        "",
        &scope.journal_records,
        &scope.resolved,
        None,
        &[],
    )?;
    let runtime_newly_retired = newly_retired_source_spans
        .iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (_, expected_coverage) =
        derive_modern_compaction_coverage(&snapshot, &runtime_newly_retired)?;
    match &event.derived_coverage {
        Some(coverage) => {
            ensure!(
                coverage == &expected_coverage,
                "context compaction derived coverage does not exactly match the pre-event runtime projection"
            );
            for item in &coverage.items {
                validate_retained_fact_text(&item.retained_text)?;
            }
            // Older records duplicated typed coverage in a summary suffix. New
            // records keep one source of truth in `derived_coverage`.
            if event.summary.contains(RETAINED_FACTS_MARKER) {
                validate_retained_facts_suffix(&event.summary, coverage)?;
            }
        }
        None => {
            ensure!(
                expected_coverage.items.is_empty(),
                "legacy context compaction event omits required modern derived coverage"
            );
            ensure!(
                !event.summary.contains(RETAINED_FACTS_MARKER),
                "context compaction summary contains retained facts without coverage"
            );
        }
    }
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
#[cfg(test)]
#[path = "transcript_projection/tests.rs"]
mod tests;
