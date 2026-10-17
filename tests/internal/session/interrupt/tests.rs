use super::*;
use crate::agent::TurnStartedEvent;
use crate::request_builder::HistoryToolCall;
use crate::transcript::{ROOT_CONTEXT_BRANCH_ID, TranscriptAssistantTurn, TranscriptEvent};
use serde_json::json;

fn turn_started(turn_id: u64) -> TranscriptEvent {
    TranscriptEvent::TurnStarted(TurnStartedEvent { turn_id })
}

fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "test-session".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

#[test]
fn committed_batch_without_started_audit_is_unfinished() {
    let records = vec![
        record(1, turn_started(1)),
        record(
            2,
            TranscriptEvent::AssistantToolCallBatch {
                text: None,
                reasoning_content: None,
                reasoning_wire: None,
                calls: vec![HistoryToolCall {
                    call_id: "batch-call".into(),
                    name: "agent__oracle".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ),
    ];

    assert_eq!(
        unfinished_tool_calls_in_active_turn(&records),
        vec![("batch-call".into(), "agent__oracle".into())]
    );
}

#[test]
fn v2_assistant_turn_without_started_audit_is_unfinished() {
    let records = vec![
        record(1, turn_started(1)),
        record(
            2,
            TranscriptEvent::AssistantTurn(TranscriptAssistantTurn {
                text: Some("working".into()),
                reasoning_content: Some("inspect".into()),
                replay: None,
                calls: vec![HistoryToolCall {
                    call_id: "v2-call".into(),
                    name: "agent__oracle".into(),
                    arguments_json: "{}".into(),
                }],
            }),
        ),
    ];

    assert_eq!(
        unfinished_tool_calls_in_active_turn(&records),
        vec![("v2-call".into(), "agent__oracle".into())]
    );
}

#[test]
fn started_subagent_without_result_is_unfinished() {
    let records = vec![
        record(1, turn_started(1)),
        record(
            2,
            TranscriptEvent::SubagentStarted {
                run_id: "run-1".into(),
                parent_session_id: "test-session".into(),
                parent_run_id: "turn-1".into(),
                child_session_id: "child-1".into(),
                agent_name: "oracle".into(),
                summary: "inspect".into(),
                pool_ordinal: 1,
            },
        ),
    ];

    let unfinished = unfinished_subagent_runs_in_active_turn(&records);
    assert_eq!(unfinished.len(), 1);
    assert_eq!(unfinished[0].run_id, "run-1");
    assert_eq!(unfinished[0].child_session_id, "child-1");
    assert_eq!(unfinished[0].agent_name, "oracle");
}
