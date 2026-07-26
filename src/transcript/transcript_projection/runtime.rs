use crate::context_view::{ContextViewProjection, project_context_view_unvalidated};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::analyze_history_items;
use crate::request_builder::HistoryItem;
use crate::runtime_context::{
    FoldedOutputReference, FrameVisibility, PromptContributorKind, PromptContributorPlaceholder,
    RuntimeChildSession, RuntimeFrame, RuntimeFrameId, RuntimeFrameIdSeed, RuntimeFrameKind,
    RuntimeFrameProvenance, RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use std::collections::BTreeSet;

use super::branch_parent_id;
use super::history::{
    HistoryProjectionEntry, HistoryProjectionOrigin, active_turn_segment_from_lifecycle_records,
    checkpoint_spans_to_compaction, restore_history_projection,
};
use super::{ResolvedBranchContext, context_scope_revision, replay_context_tree};

pub(super) fn runtime_snapshot_from_resolved_context_unbound(
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
    if let Some((turn_id, segment_id)) =
        active_turn_segment_from_lifecycle_records(&resolved.records)
    {
        snapshot = snapshot
            .with_current_turn_id(turn_id)
            .with_current_segment_id(segment_id);
    }

    let projection_records = runtime_projection_records(all_records, resolved);
    let context_tree = replay_context_tree(&projection_records)?;
    // `projection_records` is selected content, not a standalone journal. The
    // caller validates the complete journal and selected branch before this
    // branch-aware projection is built.
    let mut context_view = project_context_view_unvalidated(&projection_records)?;
    let history_entries = restore_history_projection(&resolved.records);
    let retired_source_spans = compacted_source_spans(&history_entries);
    context_view.apply_retired_spans(&retired_source_spans);
    let evidence = crate::evidence::restore_evidence_records(&resolved.records)?
        .into_iter()
        .filter(|record| !evidence_is_retired(record, &retired_source_spans))
        .collect::<Vec<_>>();

    snapshot.active_context.parent_branch_id = branch_parent_id(all_records, &resolved.branch_id)?;
    snapshot.active_context.active_node_id = context_tree
        .active_node_id()
        .map(|node_id| node_id.as_str().to_string());
    snapshot.active_context.open_detail_block_id = context_view.provider_open_detail_block_id();
    snapshot.active_context.visible_block_ids = context_view.provider_visible_block_ids();
    snapshot.active_context.pinned_block_ids = context_view.provider_pinned_block_ids();
    snapshot.set_context_tree(context_tree.clone());
    snapshot.set_context_view(context_view.clone());
    snapshot.set_evidence(evidence.clone());

    let history_frame_ids = append_history_frames(&mut snapshot, &history_entries);
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
        snapshot.current_segment_id,
    )?);
    Ok(snapshot)
}

fn evidence_is_retired(record: &EvidenceRecord, retired_spans: &[SourceSpan]) -> bool {
    let source_sequence = match &record.source {
        crate::evidence::EvidenceSource::Transcript { sequence } => Some(*sequence),
        _ => None,
    };
    retired_spans.iter().any(|span| {
        span.start_sequence <= record.sequence && record.sequence <= span.end_sequence
            || source_sequence.is_some_and(|sequence| {
                span.start_sequence <= sequence && sequence <= span.end_sequence
            })
    })
}

pub(super) fn runtime_projection_records(
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

fn compacted_source_spans(entries: &[HistoryProjectionEntry]) -> Vec<SourceSpan> {
    super::history::merge_source_spans(
        entries
            .iter()
            .filter(|entry| entry.origin == HistoryProjectionOrigin::CompactionSummary)
            .flat_map(|entry| entry.source_spans.iter().copied()),
    )
}

fn append_history_frames(
    snapshot: &mut RuntimeSnapshot,
    entries: &[HistoryProjectionEntry],
) -> Vec<RuntimeFrameId> {
    let mut frame_ids = Vec::with_capacity(entries.len());
    for (ordinal, entry) in entries.iter().enumerate() {
        let Some((kind, mut stable_key, summary)) = history_entry_frame_parts(&entry.item) else {
            continue;
        };
        if matches!(
            entry.origin,
            HistoryProjectionOrigin::LogicalCheckpointSummary
                | HistoryProjectionOrigin::LogicalCheckpointContinuation
        ) {
            stable_key = format!(
                "{}:{}",
                entry.stable_key,
                if entry.origin == HistoryProjectionOrigin::LogicalCheckpointSummary {
                    "summary"
                } else {
                    "continuation"
                }
            );
        }
        let (source, source_span) = if entry.origin == HistoryProjectionOrigin::CompactionSummary {
            (RuntimeSource::SummaryArtifact, None)
        } else {
            (
                RuntimeSource::Transcript,
                merged_runtime_source_span(&entry.source_spans),
            )
        };
        let source_id = match entry.origin {
            HistoryProjectionOrigin::LogicalCheckpointSummary => {
                Some(format!("{}:summary", entry.stable_key))
            }
            HistoryProjectionOrigin::LogicalCheckpointContinuation => {
                Some(format!("{}:continuation", entry.stable_key))
            }
            _ => None,
        };
        let provenance = runtime_provenance(source, source_span, source_id);
        let frame = RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            provenance.clone(),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source,
                ordinal: if matches!(
                    entry.origin,
                    HistoryProjectionOrigin::LogicalCheckpointSummary
                        | HistoryProjectionOrigin::LogicalCheckpointContinuation
                ) {
                    0
                } else {
                    ordinal as u32
                },
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

fn append_context_frames(
    snapshot: &mut RuntimeSnapshot,
    context_view: &ContextViewProjection,
) -> anyhow::Result<()> {
    for (ordinal, (block_id, block)) in context_view
        .all_context_blocks()
        .into_iter()
        .filter(|(_, block)| !context_view.is_tool_result_aggregate_block(block))
        .enumerate()
    {
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
    for (ordinal, metadata) in context_view
        .all_folded_outputs()
        .into_iter()
        .filter(|metadata| metadata.output_kind != "tool_result")
        .enumerate()
    {
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
    if !context_view.provider_active_blocks().is_empty() {
        let retaining_frame_ids: Vec<RuntimeFrameId> = snapshot
            .frames
            .iter()
            .filter(|frame| frame.kind == RuntimeFrameKind::ContextBlock)
            .filter(|frame| {
                let Some(block_id) = frame.provenance.source_id.as_deref() else {
                    return false;
                };
                let Ok(block_id) = crate::context_view::ContextBlockId::new(block_id) else {
                    return false;
                };
                let Some(block) = context_view.blocks.get(&block_id) else {
                    return false;
                };
                context_view.is_provider_active_block(&block_id, block)
                    && !derived_context_block_is_non_retaining(
                        context_view,
                        &block_id,
                        block,
                        frame,
                    )
                    && (block.is_protected()
                        || context_view.is_pinned_visible(&block_id)
                        || context_view.is_opened(&block_id))
            })
            .map(|frame| frame.id)
            .collect();
        let non_retaining_frame_ids = snapshot
            .frames
            .iter()
            .filter(|frame| frame.kind == RuntimeFrameKind::ContextBlock)
            .filter(|frame| {
                let Some(block_id) = frame.provenance.source_id.as_deref() else {
                    return false;
                };
                let Ok(block_id) = crate::context_view::ContextBlockId::new(block_id) else {
                    return false;
                };
                let Some(block) = context_view.blocks.get(&block_id) else {
                    return false;
                };
                context_view.is_provider_active_block(&block_id, block)
                    && derived_context_block_is_non_retaining(context_view, &block_id, block, frame)
            })
            .map(|frame| frame.id)
            .collect::<Vec<_>>();
        if !retaining_frame_ids.is_empty() {
            snapshot.push_prompt_contributor(PromptContributorPlaceholder {
                contributor_id: "context-view-active".into(),
                kind: PromptContributorKind::ContextMaterial,
                label: Some("Active context view".into()),
                provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
                retains_raw_sources: true,
                frame_ids: retaining_frame_ids,
                source_frame_ids: Vec::new(),
            });
        }
        if !non_retaining_frame_ids.is_empty() {
            snapshot.push_prompt_contributor(PromptContributorPlaceholder {
                contributor_id: "context-view-active-derived".into(),
                kind: PromptContributorKind::ContextMaterial,
                label: Some("Active derived context".into()),
                provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
                retains_raw_sources: false,
                frame_ids: non_retaining_frame_ids,
                source_frame_ids: Vec::new(),
            });
        }
    }
    if !evidence.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "evidence".into(),
            kind: PromptContributorKind::Evidence,
            label: Some("Evidence".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            // Historical evidence may co-retire only when compaction emits a
            // deterministic coverage item for its exact source span.
            retains_raw_sources: false,
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| {
                    frame.kind == RuntimeFrameKind::Metadata
                        && frame.provenance.source == RuntimeSource::Transcript
                        && frame
                            .provenance
                            .source_id
                            .as_deref()
                            .is_some_and(|id| evidence.iter().any(|record| record.id == id))
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
            // Summary artifacts are coverage/traceability only. Marking them as
            // retaining raw sources incorrectly joined their frame ids into the
            // protected set and blocked co-retirement under request pressure.
            retains_raw_sources: false,
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
            // Folded output is preserved only through deterministic semantic
            // coverage; opaque/truncated output rejects candidate preparation.
            retains_raw_sources: false,
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
            retains_raw_sources: true,
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

fn derived_context_block_is_non_retaining(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
    block: &crate::context_view::ContextBlock,
    frame: &RuntimeFrame,
) -> bool {
    // Only kinds with a typed derived-coverage encoding can be treated as
    // semantic-only. Other active materials still retain raw sources until
    // their source span is fully retired and the projection is co-retired.
    matches!(
        block.kind,
        crate::context_view::ContextBlockKind::CurrentUserRequirement
            | crate::context_view::ContextBlockKind::FileWriteFact
            | crate::context_view::ContextBlockKind::TestResult
    ) && frame.provenance.source_span.is_some()
        && !context_view.is_pinned_visible(block_id)
        && !context_view.is_opened(block_id)
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

fn merged_runtime_source_span(spans: &[SourceSpan]) -> Option<SourceSpan> {
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

fn protected_history_frame_ids(
    entries: &[HistoryProjectionEntry],
    frame_ids: &[RuntimeFrameId],
    current_turn_id: Option<u64>,
    current_segment_id: Option<u64>,
) -> anyhow::Result<Vec<RuntimeFrameId>> {
    let history = entries
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    analyze_history_items(&history, None)?;
    let active = current_turn_id.zip(current_segment_id);
    Ok(entries
        .iter()
        .zip(frame_ids)
        .filter_map(|(entry, frame_id)| {
            (Some((entry.turn_id?, entry.segment_id?)) == active).then_some(*frame_id)
        })
        .collect())
}

#[cfg(test)]
pub(super) fn snapshot_for_context_view_for_test(
    context_view: &ContextViewProjection,
) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot::new("main");
    snapshot.active_context.visible_block_ids = context_view.provider_visible_block_ids();
    append_context_frames(&mut snapshot, context_view).expect("context frames");
    append_folded_output_refs(&mut snapshot, context_view).expect("folded frames");
    append_prompt_contributors(&mut snapshot, context_view, &[], &[]).expect("prompt contributors");
    snapshot
}
