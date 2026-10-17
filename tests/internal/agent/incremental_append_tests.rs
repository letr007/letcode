use super::*;

fn evidence(id: &str) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        sequence: 1,
        timestamp_ms: 0,
        evidence_kind: crate::evidence::EvidenceKind::Decision,
        title: "decision".into(),
        summary: "preserve evidence".into(),
        detail: Some("metadata".into()),
        source: EvidenceSource::Transcript { sequence: 1 },
        tags: vec!["phase1".into()],
    }
}

#[test]
fn assistant_turn_conversion_preserves_reasoning_and_replay_without_tool_calls() {
    let item = HistoryItem::AssistantTurn {
        text: Some("answer".into()),
        reasoning_content: Some("reasoning".into()),
        replay: crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
            r#"[{"type":"thinking","signature":"signed"}]"#,
        ),
        calls: Vec::new(),
    };

    assert_eq!(protocol_frame_item_from_history_item(&item), item);
}

#[test]
fn incremental_agent_append_matches_cold_rebuild_with_metadata_and_evidence() {
    let base = vec![
        HistoryItem::user("older request"),
        HistoryItem::assistant("older answer"),
    ];
    let full_history = vec![
        base[0].clone(),
        base[1].clone(),
        HistoryItem::user("current request"),
        HistoryItem::AssistantTurn {
            text: Some("working".into()),
            reasoning_content: Some("reasoning".into()),
            replay: crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
                r#"{"signature":"wire"}"#,
            ),
            calls: vec![
                HistoryToolCall {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"a"}"#.into(),
                },
                HistoryToolCall {
                    call_id: "call-2".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"b"}"#.into(),
                },
            ],
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: r#"{"result":"b"}"#.into(),
            images: Vec::new(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: r#"{"result":"a"}"#.into(),
            images: Vec::new(),
        },
        HistoryItem::assistant("done"),
    ];

    let mut incremental = Agent::new("m1", 4, 4);
    incremental.set_history_for_test(base.clone());
    incremental.runtime_snapshot.context_scope_revision = 7;
    incremental.runtime_snapshot.current_segment_id = Some(11);
    incremental.runtime_snapshot.current_turn_id = Some(3);
    incremental.turn.current_turn_start_index = Some(base.len());
    for item in full_history.iter().skip(base.len()) {
        incremental
            .append_history_item(item.clone())
            .expect("incremental append should succeed");
    }
    incremental
        .add_evidence(evidence("ev-1"))
        .expect("evidence append should succeed");

    let mut cold = Agent::new("m1", 4, 4);
    cold.runtime_snapshot.context_scope_revision = 7;
    cold.runtime_snapshot.current_segment_id = Some(11);
    cold.runtime_snapshot.current_turn_id = Some(3);
    cold.turn.current_turn_start_index = Some(base.len());
    cold.set_history_for_test(full_history);
    cold.add_evidence(evidence("ev-1"))
        .expect("cold evidence append should succeed");

    assert_eq!(incremental.runtime_snapshot, cold.runtime_snapshot);
    assert_eq!(
        incremental.active_history_items(),
        cold.active_history_items()
    );
    assert_eq!(incremental.runtime_snapshot.leaf_sequence, Some(7));
    assert_eq!(
        incremental
            .runtime_snapshot
            .compaction
            .turn_protected_frame_ids,
        cold.runtime_snapshot.compaction.turn_protected_frame_ids
    );
    assert_eq!(incremental.evidence(), cold.evidence());
}
