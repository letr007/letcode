use crate::protocol_frames::{
    ProtocolTranscript, ToolCallGroupStatus, analyze_history_items,
    canonical_compaction_boundary_with_transcript,
};
use crate::request_builder::{HistoryItem, estimate_history_item_tokens};
use anyhow::Result;
#[cfg(test)]
use anyhow::bail;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnCut {
    pub cut_end: usize,
    pub prefix: Vec<HistoryItem>,
    pub previous_summary: Option<String>,
    /// The compacted prefix belongs to a turn that remains active in the tail.
    pub split_active_turn: bool,
}

/// Select a compactable prefix while retaining a bounded recent token tail.
///
/// A long active turn is not protected wholesale: completed tool-call groups in
/// its prefix may retire, but the retained tail always begins at a canonical
/// complete-tool boundary.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn plan_turn_cut(
    history: &[HistoryItem],
    turn_start: Option<usize>,
    preserve_recent_tokens: u64,
) -> Result<Option<TurnCut>> {
    let transcript = analyze_history_items(history, turn_start)?;
    plan_turn_cut_with_transcript(history, turn_start, preserve_recent_tokens, &transcript)
}

/// Same as [`plan_turn_cut`], but reuses an already-computed transcript analysis.
pub(crate) fn plan_turn_cut_with_transcript(
    history: &[HistoryItem],
    turn_start: Option<usize>,
    preserve_recent_tokens: u64,
    transcript: &ProtocolTranscript,
) -> Result<Option<TurnCut>> {
    if history.is_empty() {
        return Ok(None);
    }

    let summary_index = history
        .iter()
        .rposition(|item| matches!(item, HistoryItem::ContextSummary { .. }));
    let base_start = summary_index.map(|i| i + 1).unwrap_or(0);
    let previous_summary = summary_index.and_then(|i| match &history[i] {
        HistoryItem::ContextSummary { text } => Some(text.clone()),
        _ => None,
    });

    // A full compaction has no retained raw suffix and therefore no durable
    // anchor. A nonzero budget retains as much recent history as it can.
    let mut requested_boundary = history.len();
    if preserve_recent_tokens > 0 {
        let mut kept = 0u64;
        while requested_boundary > base_start {
            let cost = estimate_history_item_tokens(&history[requested_boundary - 1]);
            if kept > 0 && kept.saturating_add(cost) > preserve_recent_tokens {
                break;
            }
            kept = kept.saturating_add(cost);
            requested_boundary -= 1;
        }
    }

    // If the retained tail covers all history, there is no safe reduction;
    // otherwise canonicalize left to avoid orphaned tools.
    if requested_boundary <= base_start {
        return Ok(None);
    }
    let mut cut_end =
        canonical_compaction_boundary_with_transcript(transcript, requested_boundary)?;
    if let Some(first_incomplete) = transcript
        .tool_call_groups
        .iter()
        .filter(|group| group.status == ToolCallGroupStatus::Incomplete)
        .map(|group| group.assistant_index)
        .min()
    {
        cut_end = cut_end.min(first_incomplete);
    }
    if cut_end <= base_start {
        return Ok(None);
    }

    let split_active_turn = turn_start.is_some_and(|index| index < cut_end);
    let prefix = history[base_start..cut_end].to_vec();

    Ok(Some(TurnCut {
        cut_end,
        prefix,
        previous_summary,
        split_active_turn,
    }))
}

#[cfg(test)]
pub(crate) fn compose_with_summary(
    summary_text: impl Into<String>,
    history: &[HistoryItem],
    cut_end: usize,
) -> Result<Vec<HistoryItem>> {
    let cut_end = cut_end.min(history.len());
    if cut_end == 0 {
        bail!("compact cut_end must be > 0");
    }
    let mut out = Vec::with_capacity(1 + history.len().saturating_sub(cut_end));
    out.push(HistoryItem::context_summary(summary_text));
    out.extend_from_slice(&history[cut_end..]);
    Ok(out)
}

#[cfg(test)]
#[path = "../../tests/internal/agent/history_compact/tests.rs"]
mod tests;
