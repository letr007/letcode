use anyhow::Result;

use crate::runtime_context::RuntimeActiveContext;
use crate::transcript::transcript_projection;

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
