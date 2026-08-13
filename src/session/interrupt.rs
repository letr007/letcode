use std::sync::{Arc, Mutex};

use crate::transcript::{
    TranscriptEvent, TranscriptRecord, TranscriptRecorder, read_records, transcript_projection,
};

fn active_turn_start_index(records: &[TranscriptRecord]) -> Option<usize> {
    let mut active_turn = None;
    for (index, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::TurnStarted(event) => active_turn = Some((event.turn_id, index)),
            TranscriptEvent::TurnInterrupted { turn_id }
                if active_turn.is_none()
                    || turn_id.is_none()
                    || active_turn.map(|(id, _)| id) == *turn_id =>
            {
                active_turn = None;
            }
            TranscriptEvent::TurnFinalized(event)
                if active_turn.map(|(id, _)| id) == Some(event.turn_id) =>
            {
                active_turn = None;
            }
            _ => {}
        }
    }
    active_turn.map(|(_, index)| index)
}

/// Returns tool calls started by the currently active turn that have not reached
/// a terminal transcript event.
pub(crate) fn unfinished_tool_calls_in_active_turn(
    records: &[TranscriptRecord],
) -> Vec<(String, String)> {
    let Some(start_index) = active_turn_start_index(records) else {
        return Vec::new();
    };

    let mut unfinished = Vec::new();
    for record in &records[start_index + 1..] {
        match &record.event {
            TranscriptEvent::AssistantToolCallBatch { calls, .. } => {
                for call in calls {
                    if !unfinished.iter().any(|(id, _)| id == &call.call_id) {
                        unfinished.push((call.call_id.clone(), call.name.clone()));
                    }
                }
            }
            TranscriptEvent::ToolCallStarted { call_id, name, .. }
                if !unfinished.iter().any(|(id, _)| id == call_id) =>
            {
                unfinished.push((call_id.clone(), name.clone()));
            }
            TranscriptEvent::ToolCallFinished { call_id, .. }
            | TranscriptEvent::ToolCallCancelled { call_id, .. } => {
                unfinished.retain(|(id, _)| id != call_id);
            }
            _ => {}
        }
    }
    unfinished
}

pub(crate) struct UnfinishedSubagentRun {
    pub run_id: String,
    pub parent_session_id: String,
    pub parent_run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
}

pub(crate) fn unfinished_subagent_runs_in_active_turn(
    records: &[TranscriptRecord],
) -> Vec<UnfinishedSubagentRun> {
    let Some(start_index) = active_turn_start_index(records) else {
        return Vec::new();
    };

    let mut unfinished = Vec::new();
    for record in &records[start_index + 1..] {
        match &record.event {
            TranscriptEvent::SubagentStarted {
                run_id,
                parent_session_id,
                parent_run_id,
                child_session_id,
                agent_name,
                ..
            } if !unfinished
                .iter()
                .any(|run: &UnfinishedSubagentRun| run.run_id == *run_id) =>
            {
                unfinished.push(UnfinishedSubagentRun {
                    run_id: run_id.clone(),
                    parent_session_id: parent_session_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: agent_name.clone(),
                });
            }
            TranscriptEvent::SubagentResult { run_id, .. } => {
                unfinished.retain(|run| run.run_id != *run_id);
            }
            _ => {}
        }
    }
    unfinished
}

/// Projects the recorder's current context branch before locating unfinished
/// tool calls in its active turn.
pub(crate) fn unfinished_current_active_turn_tool_calls(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Vec<(String, String)> {
    transcript
        .lock()
        .ok()
        .and_then(|recorder| {
            let cursor = transcript_projection::SessionContextCursor {
                branch_id: recorder.current_context_branch_id().map(str::to_string),
                leaf_sequence: None,
            };
            let records = read_records(recorder.path()).ok()?;
            let session_id = recorder.session_id().to_string();
            let snapshot =
                transcript_projection::build_session_context_snapshot(session_id, records, cursor)
                    .ok()?;
            Some(unfinished_tool_calls_in_active_turn(&snapshot.records))
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::TurnStartedEvent;
    use crate::request_builder::HistoryToolCall;
    use crate::transcript::{ROOT_CONTEXT_BRANCH_ID, TranscriptEvent};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn turn_started(turn_id: u64) -> TranscriptEvent {
        TranscriptEvent::TurnStarted(TurnStartedEvent {
            turn_id,
            intent: "test".into(),
            directive: "test active turn".into(),
            validation_reminder: String::new(),
        })
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

    #[test]
    fn current_context_branch_uses_its_active_turn() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-interrupt-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&base_dir).expect("create transcript"),
        ));
        let mut recorder = transcript.lock().expect("lock transcript");
        recorder
            .record_turn_started(TurnStartedEvent {
                turn_id: 1,
                intent: "test".into(),
                directive: "root turn".into(),
                validation_reminder: String::new(),
            })
            .expect("start root turn");
        recorder
            .record_tool_call_started("root-call", "shell__exec", json!({}))
            .expect("start root tool");
        let root_leaf_sequence = read_records(recorder.path())
            .expect("read root records")
            .last()
            .expect("root turn exists")
            .sequence;
        recorder
            .record_context_branch_created(
                "branch-a",
                ROOT_CONTEXT_BRANCH_ID,
                root_leaf_sequence,
                None,
            )
            .expect("create branch");
        recorder
            .record_context_checkout("branch-a", root_leaf_sequence)
            .expect("checkout branch");
        recorder.set_current_context_branch_id(Some("branch-a".into()));
        recorder
            .record_turn_started(TurnStartedEvent {
                turn_id: 2,
                intent: "test".into(),
                directive: "branch turn".into(),
                validation_reminder: String::new(),
            })
            .expect("start branch turn");
        recorder
            .record_tool_call_started("branch-call", "fs__read", json!({}))
            .expect("start branch tool");
        drop(recorder);

        assert_eq!(
            unfinished_current_active_turn_tool_calls(&transcript),
            vec![("branch-call".into(), "fs__read".into())]
        );
    }
}
