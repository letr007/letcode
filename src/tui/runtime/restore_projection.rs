use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::runtime_context::RuntimeActiveContext;
use crate::transcript::transcript_projection;

use crate::tui::events::TokenUsageEvent;

pub(super) fn runtime_context_from_records(
    records: &[crate::transcript::TranscriptRecord],
    session_id: &str,
    branch_id: Option<&str>,
) -> Result<RuntimeActiveContext> {
    let snapshot = transcript_projection::project_runtime_restore_snapshot(
        session_id.to_string(),
        records.to_vec(),
        transcript_projection::SessionContextCursor {
            branch_id: branch_id.map(str::to_string),
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot;
    RuntimeActiveContext::try_from(&snapshot)
}

pub(super) fn project_runtime_restore_snapshot_with_children(
    session_id: &str,
    records: Vec<crate::transcript::TranscriptRecord>,
    cursor: transcript_projection::SessionContextCursor,
    sessions_dir: &std::path::Path,
) -> Result<transcript_projection::RuntimeRestoreSnapshot> {
    crate::session::project_runtime_restore_snapshot_with_children(
        session_id,
        records,
        cursor,
        sessions_dir,
    )
}

/// Session usage is a fresh estimate of the restored request. Response and
/// cache accounting is not persisted in transcripts, so it must not cross a
/// session boundary.
pub(super) fn restored_session_token_usage<C>(
    agent: &Agent<C>,
    model_id: &str,
    runtime_snapshot: &crate::runtime_context::RuntimeSnapshot,
) -> Result<TokenUsageEvent>
where
    C: Config,
{
    let usage = agent.candidate_session_token_usage(model_id, runtime_snapshot)?;
    Ok(TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    ))
}

pub(super) fn restored_messages_from_protocol_frames(
    protocol_frames: &[crate::protocol_frames::ProtocolFrame],
) -> Vec<crate::agent::ConversationMessage> {
    crate::protocol_frames::history_items_from_frames(protocol_frames)
        .into_iter()
        .filter_map(|item| match item {
            crate::request_builder::HistoryItem::ContextSummary { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Summary,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::UserMessage { content } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: content.display_text(),
                })
            }
            crate::request_builder::HistoryItem::InternalContinuation { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantText { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantToolCalls { text, .. } => {
                text.map(|content| crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content,
                })
            }
            crate::request_builder::HistoryItem::ToolOutput { .. } => None,
        })
        .collect()
}
