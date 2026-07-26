use crate::protocol_frames::ProtocolFrame;
use crate::runtime_context::{FrameVisibility, RuntimeFrame, RuntimeSnapshot};

use super::{HistoryAdapterProjection, HistoryItem};

/// Provider prompt material is history-only.
///
/// ContextView and tree projections remain available for TUI and tool
/// addressing, but they must not inject synthetic prelude or history
/// prefix frames into the request planner.
pub(super) fn runtime_context_history_adapter(
    _snapshot: &RuntimeSnapshot,
    _history: &[HistoryItem],
    _protected_start_index: usize,
) -> HistoryAdapterProjection {
    HistoryAdapterProjection::default()
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
