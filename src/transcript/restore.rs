use super::*;
use crate::agent::{AutoContinueState, ConversationMessage, ConversationRole, TodoItem};
#[cfg(test)]
use crate::evidence::{EvidenceRecord, restore_evidence_records};
use crate::request_builder::{HistoryItem, HistoryToolCall};
#[cfg(test)]
use anyhow::{Result, anyhow, ensure};
#[cfg(test)]
use serde_json::Value;

#[cfg(test)]
pub fn restore_job_board(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Result<Vec<JobBoardEntry>> {
    transcript_projection::project_job_board(&child_sessions_dir(base_dir), parent_records)
}

#[cfg(test)]
pub(crate) fn restore_session_history(records: &[TranscriptRecord]) -> Result<Vec<HistoryItem>> {
    transcript_projection::validate_context_projection_events(records)?;
    Ok(transcript_projection::restore_session_history_projection(
        records,
    ))
}

#[cfg(test)]
pub(crate) fn restore_session_protocol_frames(
    records: &[TranscriptRecord],
) -> Result<Vec<crate::protocol_frames::ProtocolFrame>> {
    transcript_projection::restore_session_protocol_frames_projection(records)
}

#[cfg(test)]
pub(crate) fn restore_runtime_snapshot(
    records: &[TranscriptRecord],
) -> Result<crate::runtime_context::RuntimeSnapshot> {
    let session_id = records
        .first()
        .map(|record| record.session_id.clone())
        .unwrap_or_else(|| "restored-session".into());
    Ok(transcript_projection::project_runtime_restore_snapshot(
        session_id,
        records.to_vec(),
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot)
}

#[cfg(test)]
pub(crate) fn restore_compacted_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    Ok(restore_session_history(records)?
        .into_iter()
        .filter_map(history_item_to_conversation_message)
        .collect())
}

#[cfg(test)]
pub(crate) fn restore_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    restore_compacted_conversation_messages(records)
}

pub fn restore_latest_model(records: &[TranscriptRecord]) -> Option<String> {
    transcript_projection::restore_latest_model_projection(records)
}

pub fn restore_latest_expert_models(
    records: &[TranscriptRecord],
) -> indexmap::IndexMap<String, String> {
    let mut models = indexmap::IndexMap::new();
    for record in records {
        if let TranscriptEvent::ExpertModelChanged { agent_name, model } = &record.event {
            models.insert(agent_name.clone(), model.clone());
        }
    }
    models
}

#[cfg(test)]
pub(crate) fn restore_latest_expert_models_for_cursor(
    session_id: &str,
    records: &[TranscriptRecord],
    cursor: transcript_projection::SessionContextCursor,
) -> anyhow::Result<indexmap::IndexMap<String, String>> {
    let snapshot = transcript_projection::build_session_context_snapshot(
        session_id.to_string(),
        records.to_vec(),
        cursor,
    )?;
    Ok(restore_latest_expert_models(&snapshot.records))
}

pub fn restore_latest_permission_mode(records: &[TranscriptRecord]) -> Option<String> {
    transcript_projection::restore_latest_permission_mode_projection(records)
}

pub fn restore_latest_reasoning_effort(
    records: &[TranscriptRecord],
    model_id: &str,
) -> Option<crate::request_builder::ModelReasoningEffort> {
    transcript_projection::restore_latest_reasoning_effort_projection(records, model_id)
}

#[cfg(test)]
pub(crate) fn restore_session_evidence(
    records: &[TranscriptRecord],
) -> Result<Vec<EvidenceRecord>> {
    restore_evidence_records(records)
}

pub fn restore_latest_todo_snapshot(records: &[TranscriptRecord]) -> Option<Vec<TodoItem>> {
    let projection = transcript_projection::project_workflow_state(records);
    projection.has_todos.then_some(projection.state.todos)
}

pub fn restore_latest_auto_continue_state(
    records: &[TranscriptRecord],
) -> Option<AutoContinueState> {
    let projection = transcript_projection::project_workflow_state(records);
    projection
        .has_auto_continue
        .then_some(projection.state.auto_continue)
}

#[cfg(test)]
pub(crate) fn restore_max_turn_id(records: &[TranscriptRecord]) -> u64 {
    transcript_projection::restore_max_turn_id_projection(records)
}

pub(crate) fn append_history_item_from_transcript_record(
    record: &TranscriptRecord,
) -> Option<HistoryItem> {
    match &record.event {
        TranscriptEvent::UserMessage { content } => {
            Some(HistoryItem::user_content(content.clone()))
        }
        TranscriptEvent::AssistantTurn(turn) => Some(HistoryItem::AssistantTurn {
            text: turn.text.clone(),
            reasoning_content: turn.reasoning_content.clone(),
            replay: turn.replay.clone(),
            calls: turn.calls.clone(),
        }),
        TranscriptEvent::AssistantMessage { content } => {
            Some(HistoryItem::assistant(content.clone()))
        }
        TranscriptEvent::InternalContinuation { text, .. } => {
            Some(HistoryItem::internal_continuation(text.clone()))
        }
        TranscriptEvent::AssistantToolCallBatch {
            text,
            reasoning_content,
            reasoning_wire,
            calls,
        } => Some(HistoryItem::AssistantTurn {
            text: text.clone(),
            reasoning_content: reasoning_content.clone(),
            replay: reasoning_wire.as_deref().and_then(
                crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json,
            ),
            calls: calls.clone(),
        }),
        TranscriptEvent::ToolCallStarted {
            call_id,
            name,
            args,
        } => Some(HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![HistoryToolCall {
                call_id: call_id.clone(),
                name: name.clone(),
                arguments_json: args.to_string(),
            }],
        }),
        TranscriptEvent::ToolCallFinished {
            call_id, output, ..
        } => Some(HistoryItem::ToolOutput {
            call_id: call_id.clone(),
            output_json: serde_json::to_string(&output.for_text_history())
                .unwrap_or_else(|_| "null".to_string()),
            images: output.images.clone(),
        }),
        TranscriptEvent::ToolCallCancelled { .. } => None,
        TranscriptEvent::ContextExperimentReturned {
            branch_id,
            outcome,
            summary,
            next_action,
            had_writes,
            ..
        } => Some(HistoryItem::context_summary(
            format_context_experiment_return(
                branch_id,
                outcome,
                summary,
                next_action.as_deref(),
                *had_writes,
            ),
        )),
        _ => None,
    }
}

#[cfg(test)]
fn required_metadata_string(metadata: &Value, field: &str) -> Result<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("invalid pending metadata: missing string field '{field}'"))
}

#[cfg(test)]
fn optional_metadata_string(metadata: &Value, field: &str) -> Option<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

#[cfg(test)]
fn optional_metadata_u32(metadata: &Value, field: &str) -> Result<Option<u32>> {
    metadata
        .get(field)
        .map(|value| match value {
            Value::Null => Ok(None),
            Value::Number(number) => number
                .as_u64()
                .ok_or_else(|| anyhow!("invalid pending metadata: field '{field}' must be u32"))
                .and_then(|value| {
                    u32::try_from(value).map_err(|_| {
                        anyhow!("invalid pending metadata: field '{field}' exceeds u32")
                    })
                })
                .map(Some),
            _ => Err(anyhow!(
                "invalid pending metadata: field '{field}' must be u32 or null"
            )),
        })
        .transpose()
        .map(|value| value.flatten())
}

#[cfg(test)]
fn optional_metadata_u64(metadata: &Value, field: &str) -> Result<Option<u64>> {
    metadata
        .get(field)
        .map(|value| match value {
            Value::Null => Ok(None),
            Value::Number(number) => number
                .as_u64()
                .ok_or_else(|| anyhow!("invalid pending metadata: field '{field}' must be u64"))
                .map(Some),
            _ => Err(anyhow!(
                "invalid pending metadata: field '{field}' must be u64 or null"
            )),
        })
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn history_item_to_conversation_message(
    item: HistoryItem,
) -> Option<ConversationMessage> {
    match item {
        HistoryItem::ContextSummary { text } => Some(ConversationMessage {
            role: ConversationRole::Summary,
            content: text,
        }),
        HistoryItem::UserMessage { content } => Some(ConversationMessage {
            role: ConversationRole::User,
            content: content.display_text(),
        }),
        HistoryItem::InternalContinuation { text } => Some(ConversationMessage {
            role: ConversationRole::User,
            content: text,
        }),
        HistoryItem::AssistantTurn {
            text: Some(text),
            calls,
            ..
        } if calls.is_empty() => Some(ConversationMessage {
            role: ConversationRole::Assistant,
            content: text,
        }),
        HistoryItem::AssistantTurn { .. } | HistoryItem::ToolOutput { .. } => None,
    }
}
