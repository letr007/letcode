use super::{
    ResolvedBranchContext, SessionContextCursor, active_turn_segment_from_lifecycle_records,
    branch_tip_for_records, build_branch_index, checkpoint_spans_from_history,
    collect_branch_path_records, context_scope_revision, resolve_active_branch_id,
    resolve_branch_context, restore_history_projection, runtime_projection_records,
};
use crate::context_view::{self, ContextBlockKind, ContextBlockSource};
use crate::protocol_frames::analyze_history_items;
use crate::request_builder::HistoryItem;
use crate::transcript::{
    LogicalCheckpointAuditSourceV1, LogicalCheckpointEventV1, LogicalCheckpointSourceSpanV1,
    TranscriptEvent, TranscriptRecord, render_checkpoint_continuation_v1, render_checkpoint_v1,
};
use anyhow::{anyhow, ensure};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn validate_logical_checkpoint_candidate(
    session_id: &str,
    all_records: &[TranscriptRecord],
    branch_id: Option<String>,
    journal_frontier: u64,
    checkpoint_sequence: u64,
    event: &LogicalCheckpointEventV1,
) -> anyhow::Result<()> {
    ensure!(
        event.schema_version == 1,
        "logical checkpoint schema_version must be 1"
    );
    ensure!(
        valid_checkpoint_id(&event.checkpoint_id),
        "logical checkpoint id has invalid grammar"
    );
    ensure!(
        event.retained_items.len() <= 64,
        "logical checkpoint has too many retained items"
    );
    ensure!(
        checkpoint_sequence
            == journal_frontier
                .checked_add(1)
                .ok_or_else(|| anyhow!("logical checkpoint journal frontier overflow"))?,
        "logical checkpoint journal frontier is inconsistent"
    );
    ensure!(all_records.iter().all(|record| !matches!(&record.event, TranscriptEvent::LogicalCheckpoint(previous) if previous.checkpoint_id == event.checkpoint_id)), "logical checkpoint id must be session-global unique");
    let journal_records = all_records
        .iter()
        .filter(|record| record.sequence <= journal_frontier)
        .cloned()
        .collect::<Vec<_>>();
    let journal_index = build_branch_index(&journal_records)?;
    let resolved_branch_id = branch_id
        .clone()
        .unwrap_or_else(|| resolve_active_branch_id(&journal_index, None));
    let content_boundary =
        branch_tip_for_records(&journal_records, &journal_index, &resolved_branch_id)?;
    // Metadata at the journal frontier is not a content leaf. Resolve the
    // branch at its content boundary while retaining the full prefix for scope.
    let journal_scope = resolve_branch_context(
        journal_records.clone(),
        SessionContextCursor {
            branch_id: Some(resolved_branch_id.clone()),
            leaf_sequence: Some(content_boundary),
        },
    )?;
    ensure!(
        context_scope_revision(&journal_records, &journal_scope) == event.context_scope_revision,
        "logical checkpoint context scope revision does not match journal frontier"
    );
    ensure!(
        event.boundary_sequence == content_boundary,
        "logical checkpoint boundary_sequence must equal branch content boundary"
    );
    let resolved = ResolvedBranchContext {
        branch_id: resolved_branch_id,
        leaf_sequence: content_boundary,
        scope_checkout_sequence: journal_scope.scope_checkout_sequence,
        records: collect_branch_path_records(
            &journal_records,
            &journal_index,
            &journal_scope.branch_id,
            content_boundary,
        )?,
    };
    let active = active_turn_segment_from_lifecycle_records(&resolved.records)
        .ok_or_else(|| anyhow!("logical checkpoint requires an active turn and segment"))?;
    ensure!(
        active.0 == event.turn_id && active.1 == event.previous_segment_id,
        "logical checkpoint does not target the active turn segment"
    );
    ensure!(
        event.segment_id
            == event
                .previous_segment_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("logical checkpoint segment id overflow"))?,
        "logical checkpoint segment lineage is not adjacent"
    );
    let prior = resolved
        .records
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            TranscriptEvent::LogicalCheckpoint(previous) => Some(previous.checkpoint_id.as_str()),
            _ => None,
        });
    ensure!(
        prior == event.previous_checkpoint_id.as_deref(),
        "logical checkpoint previous checkpoint lineage does not match branch path"
    );
    let history = restore_history_projection(&resolved.records);
    let closed = history
        .iter()
        .filter(|entry| {
            entry.turn_id == Some(event.turn_id)
                && entry.segment_id == Some(event.previous_segment_id)
        })
        .collect::<Vec<_>>();
    ensure!(
        !closed.is_empty(),
        "logical checkpoint cannot close an empty segment"
    );
    let items = history
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    let start = history.iter().position(|entry| {
        entry.turn_id == Some(event.turn_id) && entry.segment_id == Some(event.previous_segment_id)
    });
    let protocol = analyze_history_items(&items, start)?;
    ensure!(
        protocol
            .tool_call_groups
            .iter()
            .all(|group| group.status != crate::protocol_frames::ToolCallGroupStatus::Incomplete),
        "logical checkpoint requires complete protocol groups"
    );
    let expected = checkpoint_spans_from_history(&closed);
    ensure!(
        event.covered_source_spans == expected,
        "logical checkpoint covered source spans must exactly equal the closed segment closure"
    );
    validate_checkpoint_items(event, &journal_records, &journal_scope)?;
    let rendered = render_checkpoint_v1(event)?;
    ensure!(
        rendered.len() + render_checkpoint_continuation_v1(event).len() <= 32 * 1024,
        "logical checkpoint rendered content exceeds 32 KiB"
    );
    let _ = session_id;
    Ok(())
}

/// Synthesizes and proves a checkpoint candidate against the immutable committed
/// journal.  It deliberately does not write the candidate: acknowledgement is
/// retained only for validating and replaying legacy checkpoint records.
#[cfg(test)]
pub(crate) fn prepare_logical_checkpoint_candidate(
    session_id: &str,
    all_records: &[TranscriptRecord],
    branch_id: String,
    journal_frontier: u64,
) -> anyhow::Result<LogicalCheckpointEventV1> {
    let journal_records = all_records
        .iter()
        .filter(|record| record.sequence <= journal_frontier)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        journal_records.len() == all_records.len(),
        "logical checkpoint input contains records beyond the journal frontier"
    );
    let index = build_branch_index(&journal_records)?;
    let boundary = branch_tip_for_records(&journal_records, &index, &branch_id)?;
    let scope = resolve_branch_context(
        journal_records.clone(),
        SessionContextCursor {
            branch_id: Some(branch_id.clone()),
            leaf_sequence: Some(boundary),
        },
    )?;
    let resolved = ResolvedBranchContext {
        branch_id: branch_id.clone(),
        leaf_sequence: boundary,
        scope_checkout_sequence: scope.scope_checkout_sequence,
        records: collect_branch_path_records(&journal_records, &index, &scope.branch_id, boundary)?,
    };
    let (turn_id, previous_segment_id) =
        active_turn_segment_from_lifecycle_records(&resolved.records).ok_or_else(|| {
            anyhow!("logical checkpoint requires an active, available turn and segment")
        })?;
    let segment_id = previous_segment_id
        .checked_add(1)
        .ok_or_else(|| anyhow!("logical checkpoint segment id overflow"))?;
    let history = restore_history_projection(&resolved.records);
    let closed = history
        .iter()
        .filter(|entry| {
            entry.turn_id == Some(turn_id) && entry.segment_id == Some(previous_segment_id)
        })
        .collect::<Vec<_>>();
    ensure!(
        !closed.is_empty(),
        "logical checkpoint cannot close an empty segment"
    );
    let start = history.iter().position(|entry| {
        entry.turn_id == Some(turn_id) && entry.segment_id == Some(previous_segment_id)
    });
    let history_items = history
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    let protocol = analyze_history_items(&history_items, start)?;
    ensure!(
        protocol.tool_call_groups.iter().all(|group| {
            group.status != crate::protocol_frames::ToolCallGroupStatus::Incomplete
        }),
        "logical checkpoint requires complete protocol groups"
    );
    let covered_source_spans = checkpoint_spans_from_history(&closed);
    let previous = resolved
        .records
        .iter()
        .rev()
        .find_map(|record| match &record.event {
            TranscriptEvent::LogicalCheckpoint(event) => Some((record.sequence, event)),
            _ => None,
        });
    let previous_checkpoint_id = previous.map(|(_, event)| event.checkpoint_id.clone());
    let context_scope_revision = context_scope_revision(&journal_records, &scope);
    let mut retained_items = checkpoint_retained_items(
        &journal_records,
        &scope,
        &covered_source_spans,
        turn_id,
        previous_segment_id,
        previous,
    )?;
    retained_items.sort_by(|left, right| {
        (
            left.kind.rank(),
            audit_key(&left.audit_source),
            &left.title,
            &left.detail,
        )
            .cmp(&(
                right.kind.rank(),
                audit_key(&right.audit_source),
                &right.title,
                &right.detail,
            ))
    });
    retained_items.dedup();
    ensure!(
        retained_items.len() <= 64,
        "logical checkpoint has too many retained items"
    );
    #[derive(serde::Serialize)]
    struct IdInput<'a> {
        session_id: &'a str,
        branch_id: &'a str,
        turn_id: u64,
        previous_segment_id: u64,
        segment_id: u64,
        boundary_sequence: u64,
        context_scope_revision: u64,
        previous_checkpoint_id: &'a Option<String>,
    }
    let id_input = IdInput {
        session_id,
        branch_id: &branch_id,
        turn_id,
        previous_segment_id,
        segment_id,
        boundary_sequence: boundary,
        context_scope_revision,
        previous_checkpoint_id: &previous_checkpoint_id,
    };
    let checkpoint_id = format!(
        "lcp-v1-{}",
        crate::request_builder::sha256_hex(&serde_json::to_vec(&id_input)?)
    );
    let event = LogicalCheckpointEventV1 {
        schema_version: 1,
        checkpoint_id,
        turn_id,
        previous_segment_id,
        segment_id,
        previous_checkpoint_id,
        boundary_sequence: boundary,
        context_scope_revision,
        covered_source_spans,
        retained_items,
    };
    let checkpoint_sequence = journal_frontier
        .checked_add(1)
        .ok_or_else(|| anyhow!("logical checkpoint journal frontier overflow"))?;
    validate_logical_checkpoint_candidate(
        session_id,
        &journal_records,
        Some(branch_id.clone()),
        journal_frontier,
        checkpoint_sequence,
        &event,
    )?;
    let mut candidate = journal_records;
    candidate.push(TranscriptRecord {
        session_id: session_id.to_string(),
        sequence: checkpoint_sequence,
        timestamp_ms: 0,
        context_branch_id: Some(branch_id.clone()),
        event: TranscriptEvent::LogicalCheckpoint(event.clone()),
    });
    let replay = super::project_runtime_restore_snapshot(
        session_id.to_string(),
        candidate,
        SessionContextCursor {
            branch_id: Some(branch_id),
            leaf_sequence: Some(checkpoint_sequence),
        },
        &[],
    )?;
    ensure!(
        replay.snapshot.current_turn_id == Some(turn_id)
            && replay.snapshot.current_segment_id == Some(segment_id),
        "logical checkpoint candidate replay did not advance the active segment"
    );
    let rendered = render_checkpoint_v1(&event)?;
    ensure!(
        replay.snapshot.active_history_items().iter().any(|item| {
            matches!(item, HistoryItem::ContextSummary { text } if text == rendered.as_str())
        }) && replay.snapshot.active_history_items().iter().any(|item| {
            matches!(item, HistoryItem::InternalContinuation { text } if text == render_checkpoint_continuation_v1(&event).as_str())
        }),
        "logical checkpoint candidate replay is missing summary or continuation"
    );
    Ok(event)
}

fn checkpoint_retained_items(
    journal_records: &[TranscriptRecord],
    scope: &ResolvedBranchContext,
    closure: &[LogicalCheckpointSourceSpanV1],
    turn_id: u64,
    segment_id: u64,
    previous: Option<(u64, &LogicalCheckpointEventV1)>,
) -> anyhow::Result<Vec<crate::transcript::LogicalCheckpointRetainedItemV1>> {
    let covered = |start: u64, end: u64| {
        closure
            .iter()
            .any(|span| span.start_sequence <= start && end <= span.end_sequence)
    };
    // Metadata has no history span and may be appended after the branch content
    // it describes.  Classify it only in the selected scope projection: a raw
    // journal walk would let an interleaved sibling lifecycle record change the
    // owner of selected-branch metadata.
    let projection_records = runtime_projection_records(journal_records, scope);
    let context = context_view::project_context_view_unvalidated(&projection_records)?;
    let groups = covered_call_groups(&projection_records, closure)?;
    let source_segments = native_event_segments(&projection_records);
    let mut items = Vec::new();
    let has_current_requirement = context
        .provider_active_blocks()
        .iter()
        .any(|(_, block)| block.kind == ContextBlockKind::CurrentUserRequirement);
    for (_, block) in context.provider_active_blocks() {
        // A current requirement is the canonical user requirement. Hard
        // constraints often mirror it and must not consume a second retained
        // item or create divergent instructions.
        if has_current_requirement && block.kind == ContextBlockKind::HardConstraint {
            continue;
        }
        let kind = match block.kind {
            ContextBlockKind::CurrentUserRequirement | ContextBlockKind::HardConstraint => {
                crate::transcript::LogicalCheckpointRetainedKindV1::UserRequirement
            }
            ContextBlockKind::UnresolvedError => {
                crate::transcript::LogicalCheckpointRetainedKindV1::UnresolvedError
            }
            ContextBlockKind::FileWriteFact => {
                crate::transcript::LogicalCheckpointRetainedKindV1::FileWriteFact
            }
            ContextBlockKind::TestResult => {
                crate::transcript::LogicalCheckpointRetainedKindV1::TestResult
            }
            ContextBlockKind::Permission => {
                crate::transcript::LogicalCheckpointRetainedKindV1::Permission
            }
            ContextBlockKind::CommitHash => {
                crate::transcript::LogicalCheckpointRetainedKindV1::Commit
            }
            _ => continue,
        };
        let audit_source = match &block.source {
            ContextBlockSource::TranscriptSpan {
                start_sequence,
                end_sequence,
            } => {
                let current = source_segments.get(start_sequence) == Some(&(turn_id, segment_id));
                if covered(*start_sequence, *end_sequence) {
                    if matches!(
                        kind,
                        crate::transcript::LogicalCheckpointRetainedKindV1::Permission
                            | crate::transcript::LogicalCheckpointRetainedKindV1::FileWriteFact
                            | crate::transcript::LogicalCheckpointRetainedKindV1::TestResult
                            | crate::transcript::LogicalCheckpointRetainedKindV1::UnresolvedError
                    ) {
                        validate_current_fact_provenance(
                            &projection_records,
                            *start_sequence,
                            *end_sequence,
                            kind,
                            &block.title,
                            &groups,
                            turn_id,
                        )?;
                    }
                    LogicalCheckpointAuditSourceV1::TranscriptSpan {
                        start_sequence: *start_sequence,
                        end_sequence: *end_sequence,
                    }
                } else if current
                    && matches!(
                        kind,
                        crate::transcript::LogicalCheckpointRetainedKindV1::Permission
                            | crate::transcript::LogicalCheckpointRetainedKindV1::FileWriteFact
                            | crate::transcript::LogicalCheckpointRetainedKindV1::TestResult
                            | crate::transcript::LogicalCheckpointRetainedKindV1::UnresolvedError
                    )
                {
                    // Native metadata is outside history closure. Rebind it to
                    // its exact covered assistant/result protocol group.
                    let group = validate_current_fact_provenance(
                        &projection_records,
                        *start_sequence,
                        *end_sequence,
                        kind,
                        &block.title,
                        &groups,
                        turn_id,
                    )?;
                    LogicalCheckpointAuditSourceV1::TranscriptSpan {
                        start_sequence: group.finished_sequence,
                        end_sequence: group.finished_sequence,
                    }
                } else if current {
                    return Err(anyhow!(
                        "logical checkpoint current fact '{}' is outside the closed source closure",
                        block.title
                    ));
                } else {
                    // Only facts proven to belong to an earlier turn/segment may
                    // be omitted; current facts are never silently dropped.
                    continue;
                }
            }
            ContextBlockSource::FoldedOutput { output_id } => {
                let metadata = context.folded_outputs.get(output_id).ok_or_else(|| {
                    anyhow!(
                        "logical checkpoint required fact '{}' has no folded-output metadata",
                        block.title
                    )
                })?;
                let (start, end) = (metadata.source_start_sequence, metadata.source_end_sequence);
                ensure!(
                    start
                        .zip(end)
                        .is_some_and(|(start, end)| covered(start, end)),
                    "logical checkpoint required fact '{}' cannot bind safely to the closed tool call",
                    block.title
                );
                let call_id = metadata.call_id.as_deref().ok_or_else(|| {
                    anyhow!(
                        "logical checkpoint required fact '{}' has metadata without a tool call id",
                        block.title
                    )
                })?;
                let has_call = journal_records.iter().any(|record| {
                    covered(record.sequence, record.sequence)
                        && matches!(
                            &record.event,
                            TranscriptEvent::AssistantToolCallBatch { calls, .. }
                                if calls.iter().any(|call| call.call_id == call_id)
                        )
                });
                let has_finished = journal_records.iter().any(|record| {
                    covered(record.sequence, record.sequence)
                        && matches!(
                            &record.event,
                            TranscriptEvent::ToolCallFinished { call_id: finished, .. } if finished == call_id
                        )
                });
                ensure!(
                    has_call && has_finished,
                    "logical checkpoint required fact '{}' cannot bind metadata to a covered complete tool call",
                    block.title
                );
                LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
                    output_id: output_id.clone(),
                    start_sequence: start.unwrap(),
                    end_sequence: end.unwrap(),
                }
            }
            // Facts from retired/historical material are deliberately not
            // re-homed into this segment. Current facts with an unsafe source
            // are rejected by the provenance validator above.
            _ => continue,
        };
        items.push(crate::transcript::LogicalCheckpointRetainedItemV1 {
            kind,
            title: block.title.clone(),
            detail: block.detail.clone(),
            audit_source,
        });
    }
    let inherited_workflow = match previous {
        Some((sequence, event)) if covered(sequence, sequence) => event
            .retained_items
            .iter()
            .find(|item| {
                item.kind == crate::transcript::LogicalCheckpointRetainedKindV1::WorkflowState
            })
            .map(|item| serde_json::from_str(&item.detail))
            .transpose()?,
        _ => None,
    };
    if let Some(workflow) = current_workflow_item(
        &projection_records,
        &groups,
        &source_segments,
        turn_id,
        segment_id,
        inherited_workflow,
    )? {
        // This is the sole workflow item for a candidate. A current update
        // replaces all inherited workflow state, including a reset to empty.
        items.retain(|item| {
            item.kind != crate::transcript::LogicalCheckpointRetainedKindV1::WorkflowState
        });
        items.push(workflow);
    }
    if let Some((sequence, event)) = previous {
        // Prior checkpoint content is inheritable only when that checkpoint is
        // part of the active segment closure (normally its successor summary).
        // A prior-turn checkpoint is provenance, not current retained context.
        if !covered(sequence, sequence) {
            return Ok(items);
        }
        for item in &event.retained_items {
            if item.kind == crate::transcript::LogicalCheckpointRetainedKindV1::WorkflowState
                && items.iter().any(|item| {
                    item.kind == crate::transcript::LogicalCheckpointRetainedKindV1::WorkflowState
                })
            {
                continue;
            }
            items.push(crate::transcript::LogicalCheckpointRetainedItemV1 {
                kind: item.kind,
                title: item.title.clone(),
                detail: item.detail.clone(),
                audit_source: LogicalCheckpointAuditSourceV1::TranscriptSpan {
                    start_sequence: sequence,
                    end_sequence: sequence,
                },
            });
        }
    }
    Ok(items)
}

#[derive(Debug, Clone)]
pub(super) struct CoveredCallGroup {
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) assistant_sequence: u64,
    pub(super) finished_sequence: u64,
}

/// Builds the exact covered protocol index from assistant declarations and
/// finished outputs in the closed history closure.
fn covered_call_groups(
    records: &[TranscriptRecord],
    closure: &[LogicalCheckpointSourceSpanV1],
) -> anyhow::Result<Vec<CoveredCallGroup>> {
    let covered = |sequence| {
        closure
            .iter()
            .any(|span| span.start_sequence <= sequence && sequence <= span.end_sequence)
    };
    let mut groups = Vec::new();
    for declaration in records.iter().filter(|record| covered(record.sequence)) {
        let TranscriptEvent::AssistantToolCallBatch { calls, .. } = &declaration.event else {
            continue;
        };
        for call in calls {
            let finished = records
                .iter()
                .filter(|record| covered(record.sequence))
                .filter(|record| {
                    matches!(
                        &record.event,
                        TranscriptEvent::ToolCallFinished { call_id, name, .. }
                            if call_id == &call.call_id && name == &call.name
                    )
                })
                .collect::<Vec<_>>();
            ensure!(
                finished.len() == 1,
                "logical checkpoint covered call '{}' lacks exactly one finished output",
                call.call_id
            );
            groups.push(CoveredCallGroup {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                assistant_sequence: declaration.sequence,
                finished_sequence: finished[0].sequence,
            });
        }
    }
    Ok(groups)
}

/// Classifies native metadata by the lifecycle segment active when it was
/// recorded; metadata itself intentionally has no history source span.
fn native_event_segments(records: &[TranscriptRecord]) -> BTreeMap<u64, (u64, u64)> {
    let mut active = None;
    let mut result = BTreeMap::new();
    for record in records {
        match &record.event {
            TranscriptEvent::TurnStarted(event) => active = Some((event.turn_id, 0)),
            TranscriptEvent::LogicalCheckpoint(event)
                if active.map(|pair| pair.0) == Some(event.turn_id) =>
            {
                active = Some((event.turn_id, event.segment_id))
            }
            TranscriptEvent::TurnInterrupted { turn_id }
                if active.is_none()
                    || turn_id.is_none()
                    || active.map(|pair| pair.0) == *turn_id =>
            {
                active = None
            }
            TranscriptEvent::TurnFinalized(event)
                if active.map(|pair| pair.0) == Some(event.turn_id) =>
            {
                active = None
            }
            _ => {
                if let Some(segment) = active {
                    result.insert(record.sequence, segment);
                }
            }
        }
    }
    result
}

/// A source fact may name a call only when that identifier resolves
/// unambiguously to one exact covered declaration and finished output.
pub(super) fn validate_current_fact_provenance<'a>(
    records: &[TranscriptRecord],
    start: u64,
    end: u64,
    kind: crate::transcript::LogicalCheckpointRetainedKindV1,
    title: &str,
    groups: &'a [CoveredCallGroup],
    active_turn_id: u64,
) -> anyhow::Result<&'a CoveredCallGroup> {
    use crate::transcript::LogicalCheckpointRetainedKindV1 as Kind;
    if !matches!(
        kind,
        Kind::Permission | Kind::FileWriteFact | Kind::TestResult | Kind::UnresolvedError
    ) {
        return Err(anyhow!(
            "logical checkpoint fact '{}' is not a bindable tool fact",
            title
        ));
    }
    let fact = records
        .iter()
        .find(|record| record.sequence == start && record.sequence == end)
        .ok_or_else(|| {
            anyhow!(
                "logical checkpoint fact '{}' has no exact transcript source",
                title
            )
        })?;
    let (call_id, name, after_finished) = match &fact.event {
        TranscriptEvent::PermissionDecision { call_id, tool, .. } => {
            (call_id.as_deref(), tool.as_str(), false)
        }
        TranscriptEvent::ToolExecutionSummary(summary)
            if matches!(summary.effect_kind.as_str(), "write" | "validation") =>
        {
            ensure!(
                summary.turn_id == active_turn_id,
                "logical checkpoint required fact '{}' has summary turn {} instead of active turn {}",
                title,
                summary.turn_id,
                active_turn_id
            );
            (Some(summary.call_id.as_str()), summary.name.as_str(), true)
        }
        TranscriptEvent::ToolCallFinished {
            call_id,
            name,
            ok: false,
            ..
        } if kind == Kind::UnresolvedError => (Some(call_id.as_str()), name.as_str(), true),
        TranscriptEvent::Error { .. } if kind == Kind::UnresolvedError => (None, "", false),
        _ => {
            return Err(anyhow!(
                "logical checkpoint fact '{}' has unsupported provenance",
                title
            ));
        }
    };
    let call_id = call_id.ok_or_else(|| {
        anyhow!(
            "logical checkpoint required fact '{}' is missing a call_id binding",
            title
        )
    })?;
    let matches = groups
        .iter()
        .filter(|group| {
            group.call_id == call_id
                && group.name == name
                && if after_finished {
                    group.finished_sequence <= fact.sequence
                } else {
                    group.assistant_sequence <= fact.sequence
                        && fact.sequence <= group.finished_sequence
                }
        })
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "logical checkpoint required fact '{}' cannot bind call_id '{}' to one covered complete tool group",
        title,
        call_id
    );
    Ok(matches[0])
}

fn current_workflow_item(
    records: &[TranscriptRecord],
    groups: &[CoveredCallGroup],
    source_segments: &BTreeMap<u64, (u64, u64)>,
    turn_id: u64,
    segment_id: u64,
    inherited: Option<crate::transcript::CheckpointWorkflowProjection>,
) -> anyhow::Result<Option<crate::transcript::LogicalCheckpointRetainedItemV1>> {
    let mut workflow = inherited.unwrap_or_default();
    let mut source = None;
    for record in records
        .iter()
        .filter(|record| source_segments.get(&record.sequence) == Some(&(turn_id, segment_id)))
    {
        match &record.event {
            TranscriptEvent::TodoSnapshot { items } => {
                let matches = groups
                    .iter()
                    .filter(|group| {
                        group.name == crate::tool_names::TOOL_WORKFLOW_TODOS
                            && group.assistant_sequence <= record.sequence
                            && record.sequence <= group.finished_sequence
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    matches.len() == 1,
                    "logical checkpoint todo metadata cannot bind to exactly one covered workflow control group"
                );
                if items.is_empty() {
                    // A todo reset owns only the todo component.  Preserve an
                    // auto-continue setting supplied by a prior workflow event.
                    workflow.todos.clear();
                } else {
                    workflow.todos = items.clone();
                }
                source = Some(matches[0].finished_sequence);
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                let matches = groups
                    .iter()
                    .filter(|group| {
                        group.name == crate::tool_names::TOOL_WORKFLOW_AUTO_CONTINUE
                            && group.assistant_sequence <= record.sequence
                            && record.sequence <= group.finished_sequence
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    matches.len() == 1,
                    "logical checkpoint auto-continue metadata cannot bind to exactly one covered workflow control group"
                );
                workflow.auto_continue = state.clone();
                source = Some(matches[0].finished_sequence);
            }
            _ => {}
        }
    }
    let Some(sequence) = source else {
        return Ok(None);
    };
    Ok(Some(crate::transcript::LogicalCheckpointRetainedItemV1 {
        kind: crate::transcript::LogicalCheckpointRetainedKindV1::WorkflowState,
        title: "Workflow state".to_string(),
        detail: serde_json::to_string(&workflow)?,
        audit_source: LogicalCheckpointAuditSourceV1::TranscriptSpan {
            start_sequence: sequence,
            end_sequence: sequence,
        },
    }))
}

pub(super) fn validate_logical_checkpoint_record(
    session_id: &str,
    all_records: &[TranscriptRecord],
    record: &TranscriptRecord,
    event: &LogicalCheckpointEventV1,
) -> anyhow::Result<()> {
    let journal_frontier = record
        .sequence
        .checked_sub(1)
        .ok_or_else(|| anyhow!("logical checkpoint sequence cannot be zero"))?;
    let branch_id = record
        .context_branch_id
        .clone()
        .ok_or_else(|| anyhow!("logical checkpoint must be explicitly branch-scoped"))?;
    validate_logical_checkpoint_candidate(
        session_id,
        &all_records
            .iter()
            .filter(|candidate| candidate.sequence < record.sequence)
            .cloned()
            .collect::<Vec<_>>(),
        Some(branch_id),
        journal_frontier,
        record.sequence,
        event,
    )
}

fn validate_logical_checkpoints(
    session_id: &str,
    all_records: &[TranscriptRecord],
    visible: &[TranscriptRecord],
) -> anyhow::Result<()> {
    for record in visible {
        let TranscriptEvent::LogicalCheckpoint(event) = &record.event else {
            continue;
        };
        validate_logical_checkpoint_record(session_id, all_records, record, event)?;
    }
    Ok(())
}

fn valid_checkpoint_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'.' | b'_' | b':' | b'-'))
}

fn validate_checkpoint_items(
    event: &LogicalCheckpointEventV1,
    journal_records: &[TranscriptRecord],
    journal_scope: &ResolvedBranchContext,
) -> anyhow::Result<()> {
    let audit_records = runtime_projection_records(journal_records, journal_scope);
    let covered = |start, end| {
        event
            .covered_source_spans
            .iter()
            .any(|span| span.start_sequence <= start && end <= span.end_sequence)
    };
    let mut previous = None;
    for span in &event.covered_source_spans {
        ensure!(
            span.start_sequence <= span.end_sequence,
            "logical checkpoint has inverted covered source span"
        );
        ensure!(
            previous.is_none_or(|end| end < span.start_sequence),
            "logical checkpoint spans must be ordered and non-overlapping"
        );
        previous = Some(span.end_sequence);
    }
    ensure!(
        !event.covered_source_spans.is_empty() && event.covered_source_spans.len() <= 128,
        "logical checkpoint source closure must contain 1..=128 spans"
    );
    let mut last = None;
    for item in &event.retained_items {
        ensure!(
            item.title.len() <= 256 && item.detail.len() <= 4096,
            "logical checkpoint retained item exceeds byte limit"
        );
        let key = (
            item.kind.rank(),
            audit_key(&item.audit_source),
            item.title.as_bytes(),
            item.detail.as_bytes(),
        );
        ensure!(
            last.as_ref().is_none_or(|last| last < &key),
            "logical checkpoint retained items must be canonical, ordered, and unique"
        );
        last = Some(key);
        match &item.audit_source {
            LogicalCheckpointAuditSourceV1::TranscriptSpan {
                start_sequence,
                end_sequence,
            } => {
                ensure!(
                    start_sequence <= end_sequence && covered(*start_sequence, *end_sequence),
                    "logical checkpoint transcript audit source is outside closure"
                );
                ensure!(
                    span_is_fully_visible(
                        *start_sequence,
                        *end_sequence,
                        &audit_records,
                        journal_records,
                    ),
                    "logical checkpoint transcript audit source is not branch-visible"
                );
            }
            LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
                output_id,
                start_sequence,
                end_sequence,
            } => {
                ensure!(
                    !output_id.is_empty()
                        && output_id.len() <= 128
                        && start_sequence <= end_sequence
                        && covered(*start_sequence, *end_sequence),
                    "logical checkpoint folded output audit source is invalid or outside closure"
                );
                ensure!(
                    span_is_fully_visible(
                        *start_sequence,
                        *end_sequence,
                        &audit_records,
                        journal_records,
                    ),
                    "logical checkpoint folded output audit source is not branch-visible"
                );
                let matches =
                    crate::context_view::project_context_view_unvalidated(&audit_records)?
                        .folded_outputs
                        .values()
                        .filter(|artifact| {
                            artifact.output_id == *output_id
                                && artifact.source_start_sequence == Some(*start_sequence)
                                && artifact.source_end_sequence == Some(*end_sequence)
                        })
                        .count();
                ensure!(
                    matches == 1,
                    "logical checkpoint folded output audit source must match exactly one canonical frontier artifact"
                );
            }
        }
    }
    Ok(())
}

/// A span may include metadata records, but every record that exists in the
/// interval must be on the selected content path.  Testing only overlap lets
/// sibling or future records launder an audit reference into a checkpoint.
fn span_is_fully_visible(
    start: u64,
    end: u64,
    visible: &[TranscriptRecord],
    journal_records: &[TranscriptRecord],
) -> bool {
    let visible_sequences = visible
        .iter()
        .map(|record| record.sequence)
        .collect::<BTreeSet<_>>();
    journal_records
        .iter()
        .filter(|record| start <= record.sequence && record.sequence <= end)
        .any(|record| visible_sequences.contains(&record.sequence))
        && journal_records
            .iter()
            .filter(|record| start <= record.sequence && record.sequence <= end)
            .all(|record| visible_sequences.contains(&record.sequence))
}

fn audit_key(source: &LogicalCheckpointAuditSourceV1) -> (u8, u64, u64, Vec<u8>) {
    match source {
        LogicalCheckpointAuditSourceV1::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => (0, *start_sequence, *end_sequence, Vec::new()),
        LogicalCheckpointAuditSourceV1::FoldedOutputAudit {
            output_id,
            start_sequence,
            end_sequence,
        } => (
            1,
            *start_sequence,
            *end_sequence,
            output_id.as_bytes().to_vec(),
        ),
    }
}
