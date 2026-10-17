use super::*;

type HistoryItem = ProtocolItem;
type HistoryToolCall = ProtocolToolCall;

fn tool_call(call_id: &str) -> HistoryToolCall {
    HistoryToolCall {
        call_id: call_id.into(),
        name: "fs__read".into(),
        arguments_json: r#"{"path":"src/main.rs"}"#.into(),
    }
}

#[test]
fn canonical_boundary_predicate_rejects_splits_inside_tool_groups() {
    let history = vec![
        HistoryItem::user("request"),
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("call-1")],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "{}".into(),
            images: Vec::new(),
        },
        HistoryItem::assistant("done"),
    ];
    let transcript = analyze_history_items(&history, None).expect("history is valid");

    assert!(is_canonical_compaction_boundary(&transcript, 1));
    assert!(is_canonical_compaction_boundary(&transcript, 3));
    assert!(is_canonical_compaction_boundary(&transcript, 4));
    assert!(!is_canonical_compaction_boundary(&transcript, 2));
}

#[test]
fn compaction_boundary_keeps_a_complete_tool_call_batch_atomic() {
    let history = vec![
        HistoryItem::user("question"),
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("call-1"), tool_call("call-2")],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "{}".into(),
            images: Vec::new(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: "{}".into(),
            images: Vec::new(),
        },
        HistoryItem::assistant("done"),
    ];

    assert_eq!(canonical_compaction_boundary(&history, 0).unwrap(), 0);
    assert_eq!(canonical_compaction_boundary(&history, 1).unwrap(), 1);
    assert_eq!(canonical_compaction_boundary(&history, 2).unwrap(), 1);
    assert_eq!(canonical_compaction_boundary(&history, 3).unwrap(), 1);
    assert_eq!(canonical_compaction_boundary(&history, 4).unwrap(), 4);
    assert_eq!(canonical_compaction_boundary(&history, 5).unwrap(), 5);
}

#[test]
fn incremental_append_cases_match_cold_analyzer_for_valid_tail_sequences() {
    let cases = [
        vec![
            HistoryItem::user("question"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![tool_call("case-1")],
            },
            HistoryItem::ToolOutput {
                call_id: "case-1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
        ],
        vec![
            HistoryItem::assistant("before"),
            HistoryItem::AssistantTurn {
                text: Some("tail".into()),
                reasoning_content: Some("reasoning".into()),
                replay: None,
                calls: vec![tool_call("case-2")],
            },
            HistoryItem::ToolOutput {
                call_id: "case-2".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            HistoryItem::assistant("after"),
        ],
    ];

    for history in cases {
        let full = analyze_history_items(&history, None).expect("full analysis");
        let frames = history_items_to_frames(&history);
        let mut incremental = ProtocolAppendState::empty();
        for (index, frame) in frames.iter().enumerate() {
            incremental
                .append(
                    index,
                    RuntimeFrameId::from_persisted(index as u64 + 1),
                    &frame.item,
                    None,
                )
                .expect("valid incremental append");
        }
        assert_eq!(incremental.tool_call_groups(), full.tool_call_groups);
    }
}

#[test]
fn incremental_append_matches_full_analysis_for_long_tool_history() {
    let mut history = Vec::new();
    for index in 0..128 {
        history.push(HistoryItem::user(format!("question-{index}")));
        history.push(HistoryItem::assistant(format!("answer-{index}")));
    }
    history.extend([
        HistoryItem::AssistantTurn {
            text: Some("working".into()),
            reasoning_content: Some("reasoning".into()),
            replay: crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
                r#"{"signature":"wire"}"#,
            ),
            calls: vec![tool_call("call-1"), tool_call("call-2")],
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: r#"{"result":2}"#.into(),
            images: Vec::new(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: r#"{"result":1}"#.into(),
            images: Vec::new(),
        },
        HistoryItem::assistant("final"),
    ]);

    let current_turn_start = Some(history.len() - 4);
    let full = analyze_history_items(&history, current_turn_start).expect("full analysis");
    let frames = history_items_to_frames(&history);
    let mut incremental = ProtocolAppendState::empty();
    for (index, frame) in frames.iter().enumerate() {
        incremental
            .append(
                index,
                RuntimeFrameId::from_persisted(index as u64 + 1),
                &frame.item,
                current_turn_start,
            )
            .expect("incremental append");
    }

    assert_eq!(incremental.frame_count(), history.len());
    assert_eq!(incremental.tool_call_groups(), full.tool_call_groups);
    let mut expected_protected = (history.len() - 4..history.len())
        .map(|index| RuntimeFrameId::from_persisted(index as u64 + 1))
        .collect::<Vec<_>>();
    expected_protected.sort();
    assert_eq!(
        incremental.protected_frame_ids(),
        expected_protected.as_slice()
    );
    assert!(!incremental.has_incomplete_tool_call_groups());
}

#[test]
fn incremental_append_rejects_non_tail_incomplete_groups_and_accepts_matching_outputs() {
    let history = vec![
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("historical-call")],
        },
        HistoryItem::assistant("legacy continuation"),
        HistoryItem::AssistantTurn {
            text: Some("tail working".into()),
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("tail-call")],
        },
    ];
    let frames = history_items_to_frames(&history)
        .into_iter()
        .enumerate()
        .map(|(index, mut frame)| {
            frame.runtime_frame_id = Some(RuntimeFrameId::from_persisted(index as u64 + 1));
            frame
        })
        .collect::<Vec<_>>();
    let mut incremental = ProtocolAppendState::from_frames(&frames, Some(2))
        .expect("legacy incomplete group remains representable");

    assert!(incremental.has_incomplete_tool_call_groups());
    let before = incremental.clone();
    let error = incremental
        .append(
            history.len(),
            RuntimeFrameId::from_persisted(4),
            &ProtocolFrameItem::assistant("must be rejected"),
            Some(2),
        )
        .expect_err("non-ToolOutput append must remain blocked");
    assert!(error.to_string().contains("incomplete"));
    assert_eq!(incremental, before);

    let error = incremental
        .append(
            history.len(),
            RuntimeFrameId::from_persisted(4),
            &ProtocolFrameItem::ToolOutput {
                call_id: "historical-call".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            Some(2),
        )
        .expect_err("historical incomplete groups cannot be repaired");
    assert!(error.to_string().contains("orphan tool output"));
    assert_eq!(incremental, before);

    incremental
        .append(
            history.len(),
            RuntimeFrameId::from_persisted(4),
            &ProtocolFrameItem::ToolOutput {
                call_id: "tail-call".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            Some(2),
        )
        .expect("matching tail ToolOutput is accepted despite historical state");
    assert!(incremental.has_historical_incomplete_tool_call_groups());
    assert!(incremental.has_incomplete_tool_call_groups());
    assert!(incremental.incomplete_tool_call_ids().is_empty());

    let groups = incremental.tool_call_groups();
    assert_eq!(
        groups
            .iter()
            .map(|group| group.call_ids[0].as_str())
            .collect::<Vec<_>>(),
        vec!["historical-call", "tail-call"]
    );

    let error = incremental
        .append(
            history.len() + 1,
            RuntimeFrameId::from_persisted(5),
            &ProtocolFrameItem::assistant("still rejected"),
            Some(2),
        )
        .expect_err("historical incomplete state remains globally blocking");
    assert!(error.to_string().contains("incomplete"));
}

#[test]
fn validation_rejects_orphan_tool_output() {
    let history = vec![
        HistoryItem::user("question"),
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "{}".into(),
            images: Vec::new(),
        },
    ];

    let error = validate_history_items_complete(&history, Some(0))
        .expect_err("orphan tool output must fail");
    assert!(error.to_string().contains("orphan tool output"));
}

#[test]
fn analysis_marks_incomplete_group_as_protected() {
    let history = vec![
        HistoryItem::user("question"),
        HistoryItem::AssistantTurn {
            text: Some("working".into()),
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("call-1")],
        },
    ];

    let transcript = analyze_history_items(&history, Some(1)).expect("analysis succeeds");

    assert!(transcript.has_incomplete_tool_call_groups());
    assert_eq!(transcript.tool_call_groups.len(), 1);
    let group = &transcript.tool_call_groups[0];
    assert_eq!(group.status, ToolCallGroupStatus::Incomplete);
    assert!(group.protection.current_turn);
    assert!(group.protection.incomplete);
    assert_eq!(transcript.protected_history_indexes(), BTreeSet::from([1]));
}

#[test]
fn validation_rejects_dangling_assistant_tool_calls() {
    let history = vec![
        HistoryItem::user("question"),
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![tool_call("call-1")],
        },
    ];

    let error = validate_history_items_complete(&history, Some(1))
        .expect_err("dangling assistant tool calls must fail");
    assert!(error.to_string().contains("dangling assistant tool calls"));
}
