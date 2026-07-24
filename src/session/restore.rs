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
