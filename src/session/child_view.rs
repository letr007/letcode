//! Child/parent session view projection shared by frontends.
//!
//! Phase R extracts navigation selection and restore projection for child and
//! parent transcript viewing. Event emission remains frontend-owned.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::command::ChildNavigation;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::restore::{
    default_resume_cursor, project_runtime_restore_snapshot_with_children,
};
use crate::subagent::SubagentPool;
use crate::transcript::transcript_projection::{RuntimeRestoreSnapshot, SessionContextCursor};
use crate::transcript::{
    ChildSessionSummary, TranscriptRecord, TranscriptRecorder, read_child_session_records_allow_partial_tail,
    read_records,
};

/// Parent-session view projection (frontend maps this to SessionResumed-like UI).
pub struct ParentViewProjection {
    pub snapshot: RuntimeRestoreSnapshot,
    pub runtime_context: RuntimeActiveContext,
    pub evidence_count: usize,
}

/// Child-session view projection (frontend maps this to ChildSessionViewed).
pub struct ChildViewProjection {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub index: usize,
    pub total: usize,
    pub pool_ordinal: u32,
    pub records: Vec<TranscriptRecord>,
    pub runtime_context: RuntimeActiveContext,
}

/// List child sessions with stable pool ordinal synthesis for navigation UIs.
pub fn list_child_sessions_for_view(
    sessions_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    SubagentPool::child_sessions(sessions_dir, parent_records)
}

/// Resolve which child index to open for a navigation command.
pub fn select_child_navigation_index(
    children: &[ChildSessionSummary],
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
) -> Option<usize> {
    if children.is_empty() {
        return None;
    }
    let current_index = anchor_child_session_id.and_then(|child_session_id| {
        children
            .iter()
            .position(|child| child.child_session_id == child_session_id)
    });
    Some(match navigation {
        ChildNavigation::First => 0,
        ChildNavigation::Next => current_index
            .map(|index| (index + 1) % children.len())
            .unwrap_or(0),
        ChildNavigation::Prev => current_index
            .map(|index| {
                if index == 0 {
                    children.len() - 1
                } else {
                    index - 1
                }
            })
            .unwrap_or(children.len() - 1),
    })
}

/// Sessions directory for a live transcript recorder (parent of the jsonl path).
pub fn sessions_dir_from_transcript(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<std::path::PathBuf> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    recorder
        .path()
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("transcript path has no parent directory"))
}

/// Read the live session id and records under the transcript lock.
pub fn current_session_records(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<(String, Vec<TranscriptRecord>)> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    Ok((
        recorder.session_id().to_string(),
        read_records(recorder.path())?,
    ))
}

/// Project the current live transcript as a parent/root view restore package.
pub fn project_parent_session_view(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: impl AsRef<Path>,
) -> Result<ParentViewProjection> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder.current_context_branch_id().map(str::to_string),
        )
    };
    let snapshot = project_runtime_restore_snapshot_with_children(
        session_id,
        records,
        SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
        sessions_dir,
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    let evidence_count = snapshot.snapshot.evidence.len();
    Ok(ParentViewProjection {
        snapshot,
        runtime_context,
        evidence_count,
    })
}

/// Select and project a child session view package for navigation.
///
/// Returns `Ok(None)` when the parent has no child transcripts.
pub fn project_child_session_view(
    sessions_dir: impl AsRef<Path>,
    parent_session_id: impl Into<String>,
    parent_records: &[TranscriptRecord],
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
) -> Result<Option<ChildViewProjection>> {
    let sessions_dir = sessions_dir.as_ref();
    let parent_session_id = parent_session_id.into();
    let children = list_child_sessions_for_view(sessions_dir, parent_records);
    let Some(index) =
        select_child_navigation_index(&children, navigation, anchor_child_session_id)
    else {
        return Ok(None);
    };
    let child = &children[index];
    let records =
        read_child_session_records_allow_partial_tail(sessions_dir, &child.child_session_id)?;
    let snapshot = project_runtime_restore_snapshot_with_children(
        child.child_session_id.clone(),
        records,
        default_resume_cursor(),
        sessions_dir,
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    Ok(Some(ChildViewProjection {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        pool_ordinal: child.pool_ordinal,
        records: snapshot.records,
        runtime_context,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child(id: &str) -> ChildSessionSummary {
        ChildSessionSummary {
            parent_session_id: "parent".into(),
            parent_run_id: "run".into(),
            child_session_id: id.into(),
            agent_name: "explorer".into(),
            status: "done".into(),
            summary: String::new(),
            timestamp_ms: 0,
            pool_ordinal: 1,
        }
    }

    #[test]
    fn select_child_navigation_wraps() {
        let children = vec![child("a"), child("b"), child("c")];
        assert_eq!(
            select_child_navigation_index(&children, ChildNavigation::First, None),
            Some(0)
        );
        assert_eq!(
            select_child_navigation_index(&children, ChildNavigation::Next, Some("a")),
            Some(1)
        );
        assert_eq!(
            select_child_navigation_index(&children, ChildNavigation::Next, Some("c")),
            Some(0)
        );
        assert_eq!(
            select_child_navigation_index(&children, ChildNavigation::Prev, Some("a")),
            Some(2)
        );
        assert!(select_child_navigation_index(&[], ChildNavigation::First, None).is_none());
    }
}
