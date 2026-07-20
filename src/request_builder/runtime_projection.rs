use crate::context_view::{ContextViewProjection, is_tool_result_aggregate_output_id};
use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
use crate::runtime_context::{
    FrameVisibility, PromptContributorKind, RuntimeFrame, RuntimeFrameKind, RuntimeFrameProvenance,
    RuntimeSnapshot, RuntimeSource,
};

use super::{HistoryAdapterProjection, HistoryItem};

pub(super) fn runtime_context_history_adapter(
    snapshot: &RuntimeSnapshot,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> HistoryAdapterProjection {
    let mut sections = if snapshot.context_view == ContextViewProjection::default() {
        HistoryAdapterProjection::default()
    } else {
        super::context_view_history_adapter(&snapshot.context_view, history, protected_start_index)
    };
    for frame in &mut sections.history_prefix {
        frame.source_provenance = Some(RuntimeFrameProvenance::new(match &frame.item {
            ProtocolFrameItem::ContextSummary { text }
                if text.starts_with("[Context: Summaries]") =>
            {
                RuntimeSource::SummaryArtifact
            }
            ProtocolFrameItem::ContextSummary { text }
                if text.starts_with("[Context: Folded Outputs]") =>
            {
                RuntimeSource::FoldedOutput
            }
            _ => RuntimeSource::ContextView,
        }));
    }
    // A live snapshot can carry materialized context frames before a view
    // projection exists (for example during a runtime rebuild). Render those
    // frames directly rather than silently losing provider-visible context.
    if snapshot.context_view == ContextViewProjection::default() {
        for frame in snapshot.frames.iter().filter(|frame| {
            frame_is_provider_visible(snapshot, frame)
                && frame.protocol.is_none()
                && matches!(
                    frame.kind,
                    RuntimeFrameKind::ContextBlock | RuntimeFrameKind::Summary
                )
        }) {
            let Some(summary) = frame.summary.as_deref() else {
                continue;
            };
            sections.history_prefix.push(ProtocolFrame {
                runtime_frame_id: Some(frame.id),
                source_provenance: Some(frame.provenance.clone()),
                history_index: usize::MAX,
                item: ProtocolFrameItem::ContextSummary {
                    text: format!("[Context: Runtime Material]\n{summary}"),
                },
            });
        }
        for folded in snapshot.folded_outputs.iter().filter(|folded| {
            !is_tool_result_aggregate_output_id(&folded.output_id)
                && folded.source_span.is_none_or(|span| {
                    !snapshot
                        .compaction
                        .retired_source_spans
                        .iter()
                        .any(|retired| retired.overlaps(span))
                })
        }) {
            let mut provenance = RuntimeFrameProvenance::new(RuntimeSource::FoldedOutput);
            provenance.source_id = Some(folded.output_id.clone());
            sections.history_prefix.push(ProtocolFrame {
                runtime_frame_id: None,
                source_provenance: Some(provenance),
                history_index: usize::MAX,
                item: ProtocolFrameItem::ContextSummary {
                    text: format!(
                        "[Context: Folded Outputs]\n- output_id={} tool={} call_id={}",
                        folded.output_id,
                        folded.tool_name.as_deref().unwrap_or("-"),
                        folded.call_id.as_deref().unwrap_or("-")
                    ),
                },
            });
        }
    }
    // Standard projection contributors are represented by the dedicated sections
    // above. Everything else is the generic provider-visible contributor channel.
    for contributor in snapshot.prompt_contributors.iter().filter(|contributor| {
        !matches!(
            contributor.contributor_id.as_str(),
            "context-view-active"
                | "context-view-active-derived"
                | "evidence"
                | "summary-artifacts"
                | "folded-outputs"
                | "child-sessions"
        ) && contributor.kind != PromptContributorKind::SkillMaterial
    }) {
        if contributor.provenance.source_span.is_some_and(|span| {
            snapshot
                .compaction
                .retired_source_spans
                .iter()
                .any(|retired| retired.overlaps(span))
        }) {
            continue;
        }
        let text = contributor
            .frame_ids
            .iter()
            .filter_map(|id| {
                snapshot
                    .frames
                    .iter()
                    .find(|frame| frame.id == *id)
                    .filter(|frame| frame_is_provider_visible(snapshot, frame))
                    .and_then(|frame| frame.summary.as_deref())
            })
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() {
            continue;
        }
        sections.history_prefix.push(ProtocolFrame {
            runtime_frame_id: None,
            source_provenance: Some(contributor.provenance.clone()),
            history_index: usize::MAX,
            item: ProtocolFrameItem::ContextSummary {
                text: format!(
                    "[Context: {}]\n{}",
                    contributor
                        .label
                        .as_deref()
                        .unwrap_or(&contributor.contributor_id),
                    text
                ),
            },
        });
    }
    sections
}

pub(super) fn provider_visible_protocol_frames(snapshot: &RuntimeSnapshot) -> Vec<ProtocolFrame> {
    snapshot
        .frames
        .iter()
        .filter(|frame| frame_is_provider_visible(snapshot, frame))
        .filter_map(|frame| {
            frame.protocol.clone().map(|item| ProtocolFrame {
                runtime_frame_id: Some(frame.id),
                source_provenance: Some(frame.provenance.clone()),
                history_index: 0,
                item,
            })
        })
        .enumerate()
        .map(|(history_index, mut frame)| {
            frame.history_index = history_index;
            frame
        })
        .collect()
}

fn frame_is_provider_visible(snapshot: &RuntimeSnapshot, frame: &RuntimeFrame) -> bool {
    frame.visibility == FrameVisibility::Active
        && !snapshot.compaction.compacted_frame_ids.contains(&frame.id)
        && frame.provenance.source_span.is_none_or(|span| {
            !snapshot
                .compaction
                .retired_source_spans
                .iter()
                .any(|retired| retired.overlaps(span))
        })
}

pub(super) fn protected_start_index_for_snapshot(
    snapshot: &RuntimeSnapshot,
    frames: &[ProtocolFrame],
) -> usize {
    frames
        .iter()
        .position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| snapshot.compaction.protected_frame_ids.contains(&id))
        })
        .unwrap_or(frames.len())
}
