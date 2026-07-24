//! Session-owned restore projection helpers shared by TUI and line CLI.
//!
//! Phase L extracts the common "project restore snapshot including child
//! sessions" path. Agent restore and frontend timeline mapping remain outside.

use std::path::Path;

use anyhow::Result;

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
