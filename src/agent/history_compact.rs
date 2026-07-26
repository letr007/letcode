use crate::protocol_frames::{
    ToolCallGroupStatus, analyze_history_items, canonical_compaction_boundary,
};
use crate::request_builder::{HistoryItem, estimate_history_item_tokens};
use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnCut {
    pub cut_end: usize,
    pub preserved_user_index: Option<usize>,
    pub prefix: Vec<HistoryItem>,
    pub previous_summary: Option<String>,
}

/// Select a compactable prefix while retaining a bounded recent token tail.
///
/// A long active turn is not protected wholesale: completed tool-call groups in
/// its prefix may retire, but the retained tail always begins at a canonical
/// complete-tool boundary.
pub(crate) fn plan_turn_cut(
    history: &[HistoryItem],
    turn_start: Option<usize>,
    preserve_recent_tokens: u64,
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
    let mut cut_end = canonical_compaction_boundary(history, requested_boundary)?;
    let transcript = analyze_history_items(history, turn_start)?;
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

    let preserved_user_index = turn_start.filter(|index| {
        *index < cut_end && matches!(history.get(*index), Some(HistoryItem::UserMessage { .. }))
    });
    let prefix = history[base_start..cut_end]
        .iter()
        .enumerate()
        .filter(|(offset, _)| Some(base_start + offset) != preserved_user_index)
        .map(|(_, item)| item.clone())
        .collect();

    Ok(Some(TurnCut {
        cut_end,
        preserved_user_index,
        prefix,
        previous_summary,
    }))
}

pub(crate) fn compose_with_summary(
    summary_text: impl Into<String>,
    history: &[HistoryItem],
    cut_end: usize,
    preserved_user_index: Option<usize>,
) -> Result<Vec<HistoryItem>> {
    let cut_end = cut_end.min(history.len());
    if cut_end == 0 {
        bail!("compact cut_end must be > 0");
    }
    let preserved_user = preserved_user_index
        .filter(|index| *index < cut_end)
        .and_then(|index| history.get(index))
        .cloned();
    let mut out = Vec::with_capacity(
        1 + usize::from(preserved_user.is_some()) + history.len().saturating_sub(cut_end),
    );
    out.push(HistoryItem::context_summary(summary_text));
    if let Some(user) = preserved_user {
        out.push(user);
    }
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
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "{}".into(),
            },
        ];
        let cut = plan_turn_cut(&history, Some(2), 0)
            .expect("plan")
            .expect("older turns exist");
        assert_eq!(cut.cut_end, 5);
        assert_eq!(cut.preserved_user_index, Some(2));
        assert_eq!(
            cut.prefix,
            vec![
                history[0].clone(),
                history[1].clone(),
                history[3].clone(),
                history[4].clone(),
            ]
        );
    }

    #[test]
    fn plan_turn_cut_with_zero_budget_compacts_the_full_safe_history() {
        let history = vec![HistoryItem::user("old"), HistoryItem::assistant("recent")];

        let cut = plan_turn_cut(&history, None, 0)
            .expect("plan")
            .expect("history is compactable");

        assert_eq!(cut.cut_end, 2);
        assert_eq!(cut.prefix, history);
    }

    #[test]
    fn plan_turn_cut_retires_completed_prefix_inside_long_current_turn() {
        let history = vec![
            HistoryItem::user("current"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "{}".into(),
            },
            HistoryItem::assistant("recent tail"),
        ];
        let cut = plan_turn_cut(&history, Some(0), 1)
            .expect("plan")
            .expect("completed current-turn prefix is compactable");
        assert_eq!(cut.cut_end, 3);
        assert_eq!(cut.preserved_user_index, Some(0));
        assert_eq!(cut.prefix, history[1..3]);
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
        assert_eq!(cut.preserved_user_index, Some(0));
        assert_eq!(cut.prefix, vec![history[1].clone()]);
    }

    #[test]
    fn plan_turn_cut_stops_before_incomplete_tool_group() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::AssistantToolCalls {
                text: None,
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
    fn plan_turn_cut_includes_completed_tools_before_current_turn() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("x", "lookup")],
            },
            HistoryItem::ToolOutput {
                call_id: "x".into(),
                output_json: "{}".into(),
            },
            HistoryItem::user("current"),
        ];
        let cut = plan_turn_cut(&history, Some(3), 0)
            .expect("plan")
            .expect("cut");
        assert_eq!(cut.cut_end, 4);
        assert_eq!(cut.preserved_user_index, Some(3));
        assert_eq!(cut.prefix, history[..3]);
    }

    #[test]
    fn compose_with_summary_replaces_prefix() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant("a"),
            HistoryItem::user("current"),
        ];
        let out = compose_with_summary("sum", &history, 2, None).expect("compose");
        assert!(matches!(out[0], HistoryItem::ContextSummary { .. }));
        assert_eq!(out[1], HistoryItem::user("current"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn turn_boundary_compact_replaces_only_older_turns() {
        let history = vec![
            HistoryItem::user("old-a"),
            HistoryItem::assistant("old-b"),
            HistoryItem::user("current"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("c1", "fs__read")],
            },
            HistoryItem::ToolOutput {
                call_id: "c1".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
        ];
        let cut = plan_turn_cut(&history, Some(2), 0)
            .expect("plan")
            .expect("older turns");
        let out = compose_with_summary("sum", &history, cut.cut_end, cut.preserved_user_index)
            .expect("compose");
        assert!(matches!(&out[0], HistoryItem::ContextSummary { text } if text == "sum"));
        assert_eq!(
            out,
            vec![
                HistoryItem::context_summary("sum"),
                HistoryItem::user("current")
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
        assert_eq!(cut.prefix, history[1..3]);
        assert_eq!(cut.cut_end, 4);
        assert_eq!(cut.preserved_user_index, Some(3));
        let out = compose_with_summary("next", &history, cut.cut_end, cut.preserved_user_index)
            .expect("compose");
        assert!(matches!(&out[0], HistoryItem::ContextSummary { text } if text == "next"));
        assert_eq!(out[1], HistoryItem::user("current"));
        assert_eq!(out.len(), 2);
    }
}
