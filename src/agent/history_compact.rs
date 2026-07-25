use crate::protocol_frames::{ToolCallGroupStatus, analyze_history_items, canonical_compaction_boundary};
use crate::request_builder::{HistoryItem, estimate_history_item_tokens};
use crate::tool_names;
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnCut {
    pub cut_end: usize,
    pub prefix: Vec<HistoryItem>,
    pub previous_summary: Option<String>,
}

/// Older-turn prefix only; current turn (`turn_start..`) is never cut.
pub(crate) fn plan_turn_cut(
    history: &[HistoryItem],
    turn_start: Option<usize>,
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

    let turn_start = turn_start.unwrap_or(history.len()).min(history.len());
    if turn_start <= base_start {
        return Ok(None);
    }

    let cut_end = canonical_compaction_boundary(history, turn_start)?;
    if cut_end <= base_start {
        return Ok(None);
    }

    Ok(Some(TurnCut {
        cut_end,
        prefix: history[base_start..cut_end].to_vec(),
        previous_summary,
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
    let mut out = Vec::with_capacity(1 + history.len() - cut_end);
    out.push(HistoryItem::context_summary(summary_text));
    out.extend_from_slice(&history[cut_end..]);
    Ok(out)
}

/// Stub large completed tool outputs. `protect_recent_tokens == 0` stubs all eligible.
pub(crate) fn stub_large_tool_outputs(
    history: &mut [HistoryItem],
    protect_recent_tokens: u64,
) -> bool {
    let incomplete = incomplete_call_ids(history);
    let names = tool_names_by_call_id(history);
    let mut kept = 0u64;
    let mut changed = false;

    for index in (0..history.len()).rev() {
        let HistoryItem::ToolOutput {
            call_id,
            output_json,
            ..
        } = &history[index]
        else {
            continue;
        };
        if incomplete.contains(call_id) {
            continue;
        }
        let tool_name = names.get(call_id).map(String::as_str);
        if tool_name.is_some_and(is_skill_tool_name)
            || output_json.contains(super::COMPACTION_PRUNED_MARKER)
        {
            continue;
        }
        let cost = estimate_history_item_tokens(&history[index]);
        if kept.saturating_add(cost) <= protect_recent_tokens {
            kept = kept.saturating_add(cost);
            continue;
        }
        if output_json.chars().count() < super::COMPACTION_PRUNE_MIN_OUTPUT_CHARS {
            continue;
        }
        let HistoryItem::ToolOutput { output_json, .. } = &mut history[index] else {
            continue;
        };
        *output_json = pruned_tool_output_json(output_json, tool_name);
        changed = true;
    }
    changed
}

fn incomplete_call_ids(history: &[HistoryItem]) -> BTreeSet<String> {
    let Ok(transcript) = analyze_history_items(history, None) else {
        return BTreeSet::new();
    };
    transcript
        .tool_call_groups
        .iter()
        .filter(|g| g.status != ToolCallGroupStatus::Complete)
        .flat_map(|g| g.call_ids.iter().cloned())
        .collect()
}

fn tool_names_by_call_id(history: &[HistoryItem]) -> BTreeMap<String, String> {
    let mut names = BTreeMap::new();
    for item in history {
        if let HistoryItem::AssistantToolCalls { calls, .. } = item {
            for call in calls {
                names.insert(call.call_id.clone(), call.name.clone());
            }
        }
    }
    names
}

fn is_skill_tool_name(name: &str) -> bool {
    name == "skill"
        || name == tool_names::TOOL_SKILL
        || name.starts_with("skill__")
        || name.starts_with("skill/")
}

fn pruned_tool_output_json(output_json: &str, tool_name: Option<&str>) -> String {
    let original_chars = output_json.chars().count();
    let mut marker = serde_json::Map::new();
    marker.insert("pruned".into(), Value::Bool(true));
    marker.insert(
        "reason".into(),
        Value::String(super::COMPACTION_PRUNED_MARKER.to_string()),
    );
    marker.insert(
        "original_chars".into(),
        Value::Number(serde_json::Number::from(original_chars as u64)),
    );
    if let Some(tool_name) = tool_name {
        marker.insert("tool".into(), Value::String(tool_name.to_string()));
    }
    if serde_json::from_str::<Value>(output_json).is_err() {
        marker.insert("unparsed".into(), Value::Bool(true));
    }
    json!({ "_compaction": Value::Object(marker) }).to_string()
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
    fn plan_turn_cut_keeps_entire_current_turn() {
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
        let cut = plan_turn_cut(&history, Some(2))
            .expect("plan")
            .expect("older turns exist");
        assert_eq!(cut.cut_end, 2);
        assert_eq!(cut.prefix, history[..2]);
    }

    #[test]
    fn plan_turn_cut_returns_none_when_only_current_turn() {
        let history = vec![
            HistoryItem::user("current"),
            HistoryItem::assistant("reply"),
        ];
        assert!(plan_turn_cut(&history, Some(0)).expect("plan").is_none());
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
        let cut = plan_turn_cut(&history, Some(3))
            .expect("plan")
            .expect("cut");
        assert_eq!(cut.cut_end, 3);
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
    fn stub_large_tool_outputs_skips_incomplete_and_small() {
        let large = "x".repeat(super::super::COMPACTION_PRUNE_MIN_OUTPUT_CHARS + 8);
        let mut history = vec![
            HistoryItem::user("u"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("done", "lookup")],
            },
            HistoryItem::ToolOutput {
                call_id: "done".into(),
                output_json: format!(r#"{{"data":"{large}"}}"#),
            },
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![tool_call("pend", "lookup")],
            },
        ];
        assert!(stub_large_tool_outputs(&mut history, 0));
        let HistoryItem::ToolOutput { output_json, .. } = &history[2] else {
            panic!("expected tool output");
        };
        assert!(output_json.contains(super::super::COMPACTION_PRUNED_MARKER));
        assert!(matches!(
            history[3],
            HistoryItem::AssistantToolCalls { .. }
        ));
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
        let cut = plan_turn_cut(&history, Some(2))
            .expect("plan")
            .expect("older turns");
        let out = compose_with_summary("sum", &history, cut.cut_end).expect("compose");
        assert!(matches!(&out[0], HistoryItem::ContextSummary { text } if text == "sum"));
        assert_eq!(&out[1..], &history[2..]);
    }

    #[test]
    fn leading_summary_is_not_resummarized_as_prefix() {
        let history = vec![
            HistoryItem::context_summary("prior"),
            HistoryItem::user("old"),
            HistoryItem::assistant("old answer"),
            HistoryItem::user("current"),
        ];
        let cut = plan_turn_cut(&history, Some(3))
            .expect("plan")
            .expect("older turns");
        assert_eq!(cut.previous_summary.as_deref(), Some("prior"));
        assert_eq!(cut.prefix, history[1..3]);
        assert_eq!(cut.cut_end, 3);
        let out = compose_with_summary("next", &history, cut.cut_end).expect("compose");
        assert!(matches!(&out[0], HistoryItem::ContextSummary { text } if text == "next"));
        assert_eq!(out[1], HistoryItem::user("current"));
        assert_eq!(out.len(), 2);
    }
}
