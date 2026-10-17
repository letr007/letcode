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
