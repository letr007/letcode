use crate::transcript::{TranscriptEvent, TranscriptRecord};

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
            TranscriptEvent::AssistantTurn(turn) => {
                for call in &turn.calls {
                    if !unfinished.iter().any(|(id, _)| id == &call.call_id) {
                        unfinished.push((call.call_id.clone(), call.name.clone()));
                    }
                }
            }
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

#[cfg(test)]
#[path = "../../tests/internal/session/interrupt/tests.rs"]
mod tests;
