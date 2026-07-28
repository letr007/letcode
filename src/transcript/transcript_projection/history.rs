use crate::protocol_frames::analyze_history_items;
use crate::request_builder::HistoryItem;
use crate::runtime_context::SourceSpan;
use crate::transcript::{
    LogicalCheckpointSourceSpanV1, TranscriptEvent, TranscriptRecord,
    render_checkpoint_continuation_v1, render_checkpoint_v1,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub(super) struct HistoryProjectionEntry {
    pub(super) item: HistoryItem,
    /// The exact raw source bindings for this projected item.
    pub(super) source_spans: Vec<SourceSpan>,
    pub(super) turn_id: Option<u64>,
    pub(super) segment_id: Option<u64>,
    pub(super) origin: HistoryProjectionOrigin,
    pub(super) stable_key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HistoryProjectionOrigin {
    RawTranscript,
    CompactionSummary,
    CompactionContinuation,
    LogicalCheckpointSummary,
    LogicalCheckpointContinuation,
}

/// Normalize legacy persisted compaction boundaries by clamping, canonicalizing
/// tool-call groups, and retaining incomplete groups. Projection stays tolerant
/// so historical records remain replayable.
pub(super) fn normalize_compaction_tail_start(
    history: &[HistoryProjectionEntry],
    requested: usize,
) -> usize {
    let items = history
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    let mut boundary =
        crate::protocol_frames::canonical_compaction_boundary(&items, requested.min(items.len()))
            .expect("history projection produces protocol-valid boundaries");
    let transcript = crate::protocol_frames::analyze_history_items(&items, None)
        .expect("history projection produces protocol-valid frames");
    if let Some(first_incomplete) = transcript
        .tool_call_groups
        .iter()
        .filter(|group| group.status == crate::protocol_frames::ToolCallGroupStatus::Incomplete)
        .map(|group| group.assistant_index)
        .min()
    {
        boundary = boundary.min(first_incomplete);
    }
    boundary
}

pub(crate) fn restore_session_history_projection(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    restore_history_projection(records)
        .into_iter()
        .map(|entry| entry.item)
        .collect()
}

pub(super) fn restore_history_projection(
    records: &[TranscriptRecord],
) -> Vec<HistoryProjectionEntry> {
    let mut history: Vec<HistoryProjectionEntry> = Vec::new();
    let mut active_turn_id = None;
    let mut active_segment_id = None;
    for (index, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::TurnStarted(event) => {
                // Recorders historically append the user frame before the start
                // audit record. Associate that adjacent frame with this turn.
                if let Some(previous) = history.last_mut()
                    && matches!(previous.item, HistoryItem::UserMessage { .. })
                {
                    previous.turn_id = Some(event.turn_id);
                    previous.segment_id = Some(0);
                }
                active_turn_id = Some(event.turn_id);
                active_segment_id = Some(0);
            }
            TranscriptEvent::ContextCompaction(event) => {
                // Modern compactions carry a durable anchor from the exact
                // pre-event projection. A missing anchor means no raw suffix.
                // Legacy index records remain tolerant during replay.
                let tail_start = match &event.first_kept_entry_id {
                    Some(first_kept_entry_id) => history
                        .iter()
                        .position(|entry| entry.stable_key == *first_kept_entry_id)
                        .expect("validated modern first_kept_entry_id exists in pre-compaction projection"),
                    None if event.tail_start_index.is_none() => history.len(),
                    None => normalize_compaction_tail_start(
                        &history,
                        event.tail_start_index.expect("legacy tail_start_index is present"),
                    ),
                };
                // Legacy index records replay their historical split-turn
                // layout, including a preserved active user for checkpoint
                // continuations. Modern anchor records retire the entire prefix.
                let preserved_user_index = event
                    .tail_start_index
                    .is_some()
                    .then(|| {
                        active_turn_id
                            .and_then(|turn_id| {
                                history.iter().position(|entry| {
                                    entry.turn_id == Some(turn_id)
                                        && matches!(entry.item, HistoryItem::UserMessage { .. })
                                })
                            })
                            .filter(|index| *index < tail_start)
                    })
                    .flatten();
                let retired_spans = merge_source_spans(
                    history
                        .iter()
                        .enumerate()
                        .take(tail_start)
                        .filter(|(index, _)| Some(*index) != preserved_user_index)
                        .flat_map(|(_, entry)| entry.source_spans.iter().copied()),
                );
                let preserved_user = preserved_user_index.map(|index| history[index].clone());
                let mut compacted = Vec::with_capacity(
                    1 + usize::from(preserved_user.is_some())
                        + history.len().saturating_sub(tail_start),
                );
                compacted.push(HistoryProjectionEntry {
                    item: HistoryItem::context_summary(event.summary.clone()),
                    source_spans: retired_spans.clone(),
                    turn_id: None,
                    segment_id: None,
                    origin: HistoryProjectionOrigin::CompactionSummary,
                    stable_key: format!("compaction:{}", record.sequence),
                });
                if let Some(user) = preserved_user {
                    compacted.push(user);
                }
                if let Some(checkpoint) = &event.checkpoint {
                    // Legacy checkpoint records preserve their historical
                    // continuation replay semantics. Modern events never carry
                    // checkpoints and therefore never introduce this frame.
                    let continuation_source = vec![
                        SourceSpan::new(record.sequence, record.sequence)
                            .expect("single compaction record source span is valid"),
                    ];
                    compacted.push(HistoryProjectionEntry {
                        item: HistoryItem::internal_continuation(checkpoint.continuation.clone()),
                        source_spans: continuation_source,
                        turn_id: preserved_user_index.and_then(|_| active_turn_id),
                        segment_id: preserved_user_index.and_then(|_| active_segment_id),
                        origin: HistoryProjectionOrigin::CompactionContinuation,
                        stable_key: format!("compaction:{}:continuation", record.sequence),
                    });
                }
                compacted.extend(history.drain(tail_start..));
                history = compacted;
            }
            TranscriptEvent::LogicalCheckpoint(event) => {
                // Public compatibility restore entry points validate replacement
                // events before reaching this projection.
                let closed =
                    active_segment_id.expect("validated logical checkpoint has active segment");
                let closure = checkpoint_spans_to_compaction(&event.covered_source_spans);
                history.retain(|entry| {
                    !(entry.turn_id == active_turn_id && entry.segment_id == Some(closed))
                });
                let source = vec![
                    SourceSpan::new(record.sequence, record.sequence)
                        .expect("single record source span is valid"),
                ];
                history.push(HistoryProjectionEntry {
                    item: HistoryItem::context_summary(
                        render_checkpoint_v1(event).expect("validated logical checkpoint renders"),
                    ),
                    source_spans: source.clone(),
                    turn_id: active_turn_id,
                    segment_id: Some(event.segment_id),
                    origin: HistoryProjectionOrigin::LogicalCheckpointSummary,
                    stable_key: format!("{}:summary", event.checkpoint_id),
                });
                history.push(HistoryProjectionEntry {
                    item: HistoryItem::InternalContinuation {
                        text: render_checkpoint_continuation_v1(event),
                    },
                    source_spans: source,
                    turn_id: active_turn_id,
                    segment_id: Some(event.segment_id),
                    origin: HistoryProjectionOrigin::LogicalCheckpointContinuation,
                    stable_key: format!("{}:continuation", event.checkpoint_id),
                });
                active_segment_id = Some(event.segment_id);
                let _ = closure;
            }
            TranscriptEvent::TurnInterrupted { turn_id } => {
                if active_turn_id.is_none() || turn_id.is_none() || *turn_id == active_turn_id {
                    close_interrupted_turn(&mut history, record.sequence);
                    active_turn_id = None;
                    active_segment_id = None;
                }
            }
            TranscriptEvent::TurnFinalized(event) if event.outcome == "interrupted" => {
                if Some(event.turn_id) == active_turn_id {
                    close_interrupted_turn(&mut history, record.sequence);
                    active_turn_id = None;
                    active_segment_id = None;
                }
            }
            TranscriptEvent::TurnFinalized(event) if Some(event.turn_id) == active_turn_id => {
                active_turn_id = None;
                active_segment_id = None;
            }
            _ => append_history_projection_entry_from_transcript_record(
                &mut history,
                record,
                active_turn_id,
                active_segment_id,
            ),
        }
    }
    let cancelled_call_ids = records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::ToolCallCancelled { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    normalize_incomplete_tool_call_groups(&mut history, active_turn_id, &cancelled_call_ids);
    history
}

/// Returns the unmatched lifecycle start visible at the selected branch leaf.
/// Historical turn IDs are an allocation counter, not evidence of a live turn.
pub(super) fn active_turn_id_from_lifecycle_records(records: &[TranscriptRecord]) -> Option<u64> {
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

pub(super) fn active_turn_segment_from_lifecycle_records(
    records: &[TranscriptRecord],
) -> Option<(u64, u64)> {
    let mut active = None;
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
            _ => {}
        }
    }
    active
}

pub(super) fn checkpoint_spans_to_compaction(
    spans: &[LogicalCheckpointSourceSpanV1],
) -> Vec<SourceSpan> {
    spans
        .iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence))
        .collect::<anyhow::Result<_>>()
        .expect("validated logical checkpoint source spans")
}

pub(super) fn checkpoint_spans_from_history(
    entries: &[&HistoryProjectionEntry],
) -> Vec<LogicalCheckpointSourceSpanV1> {
    let mut spans = entries
        .iter()
        .flat_map(|entry| entry.source_spans.iter())
        .map(|span| LogicalCheckpointSourceSpanV1 {
            start_sequence: span.start_sequence,
            end_sequence: span.end_sequence,
        })
        .collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.start_sequence, span.end_sequence));
    let mut merged: Vec<LogicalCheckpointSourceSpanV1> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.start_sequence <= last.end_sequence.checked_add(1).unwrap_or(u64::MAX)
        {
            last.end_sequence = last.end_sequence.max(span.end_sequence);
        } else {
            merged.push(span);
        }
    }
    merged
}

/// Historical tool calls cannot be resumed after a process restart. Remove an
/// incomplete group from the *projection* (never from the append-only
/// transcript), retaining any assistant text as a normal assistant message.
/// This leaves subsequent user turns protocol-legal without reordering records.
fn normalize_incomplete_tool_call_groups(
    history: &mut Vec<HistoryProjectionEntry>,
    active_turn_id: Option<u64>,
    cancelled_call_ids: &BTreeSet<&str>,
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
        .collect::<BTreeSet<_>>();
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
                .collect::<BTreeSet<_>>();
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
        .collect::<BTreeMap<_, _>>();
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
                segment_id: entry.segment_id,
                origin: entry.origin,
                stable_key: entry.stable_key,
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
                    segment_id: None,
                    origin: HistoryProjectionOrigin::RawTranscript,
                    stable_key: format!("cancelled:{call_id}"),
                });
            }
        }
    }
    *history = normalized;
}

fn close_interrupted_turn(history: &mut Vec<HistoryProjectionEntry>, sequence: u64) {
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
            segment_id: None,
            origin: HistoryProjectionOrigin::RawTranscript,
            stable_key: format!("interruption-close:{sequence}"),
        });
    }
}

fn append_history_projection_entry_from_transcript_record(
    history: &mut Vec<HistoryProjectionEntry>,
    record: &TranscriptRecord,
    active_turn_id: Option<u64>,
    active_segment_id: Option<u64>,
) {
    // Committed batches are the protocol authority. Their subsequent lifecycle
    // starts are audit records only and must not create a second group.
    if let TranscriptEvent::ToolCallStarted { call_id, .. } = &record.event
        && tool_call_is_declared_by_incomplete_group(history, call_id)
    {
        return;
    }
    if let Some(item) = super::super::append_history_item_from_transcript_record(record) {
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
            segment_id: active_segment_id,
            origin: HistoryProjectionOrigin::RawTranscript,
            stable_key: format!("raw:{}", record.sequence),
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

fn source_spans_for_history_record(record: &TranscriptRecord) -> Vec<SourceSpan> {
    match &record.event {
        TranscriptEvent::UserMessage { .. }
        | TranscriptEvent::AssistantMessage { .. }
        | TranscriptEvent::AssistantToolCallBatch { .. }
        | TranscriptEvent::InternalContinuation { .. }
        | TranscriptEvent::ToolCallStarted { .. }
        | TranscriptEvent::ToolCallFinished { .. }
        | TranscriptEvent::ContextExperimentReturned { .. } => {
            vec![
                SourceSpan::new(record.sequence, record.sequence)
                    .expect("single record source span is valid"),
            ]
        }
        _ => Vec::new(),
    }
}

pub(super) fn merge_source_spans(spans: impl IntoIterator<Item = SourceSpan>) -> Vec<SourceSpan> {
    let mut spans = spans.into_iter().collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.start_sequence, span.end_sequence));
    let mut merged: Vec<SourceSpan> = Vec::new();
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
