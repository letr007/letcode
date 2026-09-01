use crate::protocol_frames::{
    ProtocolTranscript, ToolCallGroupStatus, analyze_history_items,
    canonical_compaction_boundary_with_transcript,
};
use crate::request_builder::{HistoryItem, estimate_history_item_tokens};
use anyhow::{Result, bail};

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
mod tests {
    use super::*;
    use crate::request_builder::HistoryToolCall;

    fn tool_call(id: &str, name: &str) -> HistoryToolCall {
        HistoryToolCall {
            call_id: id.into(),
            name: name.into(),
            arguments_json: "{}".into(),
        }
    }

    #[test]
    fn plan_turn_cut_keeps_recent_complete_tool_group() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("old answer"),
            HistoryItem::user("current"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
        ];
        let cut = plan_turn_cut(&history, Some(2), 0)
            .expect("plan")
            .expect("older turns exist");
        assert_eq!(cut.cut_end, history.len());
        assert!(cut.split_active_turn);
        assert_eq!(cut.prefix, history);
    }

    #[test]
    fn plan_turn_cut_with_zero_budget_compacts_the_full_safe_history() {
        let history = vec![HistoryItem::user("old"), HistoryItem::assistant("recent")];

        let cut = plan_turn_cut(&history, None, 0)
            .expect("plan")
            .expect("history is compactable");

        assert_eq!(cut.cut_end, history.len());
        assert_eq!(cut.prefix, history);
    }

    #[test]
    fn plan_turn_cut_compacts_a_single_entry_without_a_suffix() {
        let history = [HistoryItem::user("only")];
        let cut = plan_turn_cut(&history, None, 0)
            .expect("plan")
            .expect("single entry is compactable");

        assert_eq!(cut.cut_end, history.len());
        assert_eq!(cut.prefix, history);
    }

    #[test]
    fn plan_turn_cut_retires_completed_prefix_inside_long_current_turn() {
        let history = vec![
            HistoryItem::user("current"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            HistoryItem::assistant("recent tail"),
        ];
        let cut = plan_turn_cut(&history, Some(0), 1)
            .expect("plan")
            .expect("completed current-turn prefix is compactable");
        assert_eq!(cut.cut_end, 3);
        assert!(cut.split_active_turn);
        assert_eq!(cut.prefix, history[..3]);
    }

    #[test]
    fn plan_turn_cut_compacts_completed_prefix_inside_current_turn() {
        let history = vec![
            HistoryItem::user("current"),
            HistoryItem::assistant("reply"),
        ];
        let cut = plan_turn_cut(&history, Some(0), 0)
            .expect("plan")
            .expect("completed prefix is compactable");
        assert_eq!(cut.cut_end, 2);
        assert!(cut.split_active_turn);
        assert_eq!(cut.prefix, history);
    }

    #[test]
    fn plan_turn_cut_stops_before_incomplete_tool_group() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("pending", "lookup")],
            },
            HistoryItem::assistant("recent tail"),
        ];

        let cut = plan_turn_cut(&history, None, 1)
            .expect("plan")
            .expect("older safe prefix remains compactable");
        assert_eq!(cut.cut_end, 1);
        assert_eq!(cut.prefix, history[..1]);
    }

    #[test]
    fn compose_with_summary_replaces_prefix() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("a"),
            HistoryItem::user("current"),
        ];
        let out = compose_with_summary("sum", &history, 2).expect("compose");
        assert!(matches!(out[0], HistoryItem::ContextSummary { .. }));
        assert_eq!(out[1], HistoryItem::user("current"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn zero_budget_compaction_replaces_the_full_safe_history() {
        let history = vec![
            HistoryItem::user("old-a"),
            HistoryItem::assistant("old-b"),
            HistoryItem::user("current"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: r#"{"ok":true}"#.into(),
                images: Vec::new(),
            },
        ];
        let cut = plan_turn_cut(&history, Some(2), 0)
            .expect("plan")
            .expect("older turns");
        let out = compose_with_summary("sum", &history, cut.cut_end).expect("compose");
        assert!(matches!(&out[0], HistoryItem::ContextSummary { text } if text == "sum"));
        assert_eq!(out, vec![HistoryItem::context_summary("sum")]);
    }

    #[test]
    fn split_turn_compaction_retires_user_and_does_not_insert_continuation() {
        let history = vec![
            HistoryItem::user("current"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            HistoryItem::assistant("recent tail"),
        ];
        let cut = plan_turn_cut(&history, Some(0), 1)
            .expect("plan")
            .expect("completed active-turn prefix");

        let out = compose_with_summary("checkpoint", &history, cut.cut_end).expect("compose");

        assert!(cut.split_active_turn);
        assert_eq!(
            out,
            vec![
                HistoryItem::context_summary("checkpoint"),
                HistoryItem::assistant("recent tail"),
            ]
        );
    }

    #[test]
    fn leading_summary_is_not_resummarized_as_prefix() {
        let history = vec![
            HistoryItem::context_summary("prior"),
            HistoryItem::user("old"),
            HistoryItem::assistant("old answer"),
            HistoryItem::user("current"),
        ];
        let cut = plan_turn_cut(&history, Some(3), 0)
            .expect("plan")
            .expect("older turns");
        assert_eq!(cut.previous_summary.as_deref(), Some("prior"));
        assert_eq!(cut.prefix, history[1..]);
        assert_eq!(cut.cut_end, history.len());
        assert!(cut.split_active_turn);
        let out = compose_with_summary("next", &history, cut.cut_end).expect("compose");
        assert_eq!(out, vec![HistoryItem::context_summary("next")]);
    }
}
