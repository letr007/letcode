//! Session-owned restore projection helpers shared by TUI and line CLI.
//!
//! Phase L extracts the common "project restore snapshot including child
//! sessions" path. Agent restore and frontend timeline mapping remain outside.

use std::path::Path;

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::session::context_scope::{apply_prepared_context_scope, prepare_context_scope};
use crate::transcript::transcript_projection::{
    RuntimeRestoreSnapshot, SessionContextCursor, project_runtime_restore_snapshot,
};
use crate::transcript::{TranscriptRecord, list_child_sessions_for_parent};

/// Default cursor for resume: active branch tip (no explicit leaf cut).
pub fn default_resume_cursor() -> SessionContextCursor {
    SessionContextCursor {
        branch_id: None,
        leaf_sequence: None,
    }
}

/// Project a runtime restore snapshot, resolving child sessions under
/// `sessions_dir` from the first-pass projection records.
pub fn project_runtime_restore_snapshot_with_children(
    session_id: impl Into<String>,
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
    sessions_dir: impl AsRef<Path>,
) -> Result<RuntimeRestoreSnapshot> {
    let session_id = session_id.into();
    let resolved = project_runtime_restore_snapshot(
        session_id.clone(),
        records.clone(),
        cursor.clone(),
        &[],
    )?;
    let children = list_child_sessions_for_parent(sessions_dir.as_ref(), &resolved.records);
    project_runtime_restore_snapshot(session_id, records, cursor, &children)
}

/// Session-owned resume package: records + restore snapshot + open recorder
/// with legacy branch adopted. Agent restore remains the caller's job.
pub struct PreparedResume {
    pub session_id: String,
    pub records: Vec<TranscriptRecord>,
    pub snapshot: RuntimeRestoreSnapshot,
    pub recorder: crate::transcript::TranscriptRecorder,
}

/// Load records, project restore snapshot (with children), open the transcript,
/// and adopt the restored branch on the recorder.
pub fn prepare_resume_package(
    sessions_dir: impl AsRef<Path>,
    session_id: impl Into<String>,
) -> Result<PreparedResume> {
    use crate::session::lifecycle::{load_session_records, open_resume_transcript};

    let sessions_dir = sessions_dir.as_ref();
    let session_id = session_id.into();
    let records = load_session_records(sessions_dir, &session_id)?;
    let snapshot = project_runtime_restore_snapshot_with_children(
        session_id.clone(),
        records.clone(),
        default_resume_cursor(),
        sessions_dir,
    )?;
    let mut recorder = open_resume_transcript(sessions_dir, &session_id)?;
    recorder.adopt_legacy_linear_branch(&snapshot.branch_id)?;
    Ok(PreparedResume {
        session_id,
        records,
        snapshot,
        recorder,
    })
}

/// Apply a prepared resume package onto the agent: restore runtime snapshot,
/// adopt latest model when present, and sync context-scope from the recorder.
///
/// Does **not** swap the live transcript recorder — callers still own that
/// under their locking / cleanup rules.
pub fn apply_prepared_resume_to_agent<C: Config>(
    agent: &mut Agent<C>,
    prepared: &PreparedResume,
) -> Result<()> {
    agent.restore_new_session_runtime_snapshot(
        prepared.snapshot.protocol_frames.clone(),
        prepared.snapshot.snapshot.clone(),
        prepared.snapshot.max_turn_id,
    )?;
    if let Some(model) = &prepared.snapshot.latest_model {
        agent.set_model(model.clone());
    }
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(())
}

/// Timeline-facing conversation messages restored from protocol frames.
pub fn restored_messages_from_protocol_frames(
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
            crate::request_builder::HistoryItem::AssistantToolCalls { text, .. } => text.map(
                |content| crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content,
                },
            ),
            _ => None,
        })
        .collect()
}

/// Fresh token estimate for a restored session request.
///
/// Response and cache accounting are not persisted in transcripts, so they must
/// not cross a session boundary (always zeroed here).
pub fn restored_session_token_usage<C: Config>(
    agent: &Agent<C>,
    model_id: &str,
    runtime_snapshot: &crate::runtime_context::RuntimeSnapshot,
) -> Result<crate::session::event::TokenUsageEvent> {
    let usage = agent.candidate_session_token_usage(model_id, runtime_snapshot)?;
    Ok(crate::session::event::TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    ))
}

/// Build the runner event emitted after a successful resume install.
///
/// Call this **before** moving `prepared.recorder` into the live transcript slot.
pub fn session_resumed_event(
    prepared: &PreparedResume,
    runtime_context: crate::runtime_context::RuntimeActiveContext,
    token_usage: Option<crate::session::event::TokenUsageEvent>,
) -> crate::session::runner::RunnerEvent {
    let snapshot = &prepared.snapshot;
    crate::session::runner::RunnerEvent::SessionResumed {
        session_id: prepared.session_id.clone(),
        branch_id: snapshot.branch_id.clone(),
        messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
        records: snapshot.records.clone(),
        evidence_count: snapshot.snapshot.evidence.len(),
        model_id: snapshot.latest_model.clone(),
        token_usage,
        runtime_context,
    }
}
