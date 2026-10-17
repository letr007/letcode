//! Child/parent session view projection shared by frontends.
//!
//! Phase R extracts navigation selection and restore projection for child and
//! parent transcript viewing. Event emission remains frontend-owned.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::command::ChildNavigation;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::restore::{
    default_resume_cursor, project_runtime_restore_snapshot_with_children,
};
use crate::subagent::SubagentPool;
use crate::transcript::transcript_projection::project_runtime_restore_snapshot;
use crate::transcript::transcript_projection::{
    RuntimeRestoreSnapshot, SessionContextCursor, project_child_session_summaries_from_file,
};
use crate::transcript::{
    ChildSessionSummary, TranscriptRecord, TranscriptRecorder, child_sessions_dir,
    read_child_session_records_allow_partial_tail, read_records,
};

/// Streaming assistant text of a child session that is still in flight.
///
/// The journal only holds durable messages, so a child view projection reads
/// this buffer to include the child's current, unpersisted answer.
#[derive(Clone, Default)]
pub(crate) struct ChildLiveText {
    inner: Arc<Mutex<HashMap<String, String>>>,
}

impl ChildLiveText {
    pub(crate) fn append(&self, child_session_id: &str, delta: &str) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner
            .entry(child_session_id.to_string())
            .or_default()
            .push_str(delta);
    }

    pub(crate) fn get(&self, child_session_id: &str) -> Option<String> {
        self.inner.lock().ok()?.get(child_session_id).cloned()
    }

    pub(crate) fn clear(&self, child_session_id: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.remove(child_session_id);
        }
    }
}

/// Parent-session view projection (frontend maps this to SessionResumed-like UI).
pub struct ParentViewProjection {
    pub snapshot: RuntimeRestoreSnapshot,
    pub runtime_context: RuntimeActiveContext,
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
    pub in_progress_assistant_text: Option<String>,
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
        ChildNavigation::List => return None,
        ChildNavigation::Session(session_id) => {
            return children
                .iter()
                .position(|child| child.child_session_id == session_id);
        }
        ChildNavigation::Toggle | ChildNavigation::First => 0,
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
    Ok(ParentViewProjection {
        snapshot,
        runtime_context,
    })
}

/// Project a child view while discovering parent children from a streaming scan.
pub fn project_child_session_view_from_file(
    sessions_dir: impl AsRef<Path>,
    parent_session_id: impl Into<String>,
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
    live_text: &ChildLiveText,
) -> Result<Option<ChildViewProjection>> {
    let sessions_dir = sessions_dir.as_ref();
    let parent_session_id = parent_session_id.into();
    let children = project_session_members(sessions_dir, &parent_session_id)?;
    project_child_session_view_with_children(
        sessions_dir,
        parent_session_id,
        children,
        navigation,
        anchor_child_session_id,
        live_text,
    )
}

pub(crate) fn project_session_members(
    sessions_dir: &Path,
    root_session_id: &str,
) -> Result<Vec<ChildSessionSummary>> {
    let parent_path = sessions_dir.join(format!("{root_session_id}.jsonl"));
    let records = crate::transcript::read_records_allow_partial_tail(&parent_path)?;
    if crate::transcript::restore_subagents_policy(&records)?.mode
        == crate::config::SubagentMode::Swarm
    {
        let snapshot = crate::subagent::swarm::project(sessions_dir, &records)?;
        return Ok(snapshot
            .latest_jobs()
            .into_iter()
            .filter_map(|job| {
                Some(ChildSessionSummary {
                    parent_session_id: job.parent_session_id?,
                    parent_run_id: job.run_id?,
                    child_session_id: job.child_session_id?,
                    agent_name: job.role,
                    status: job.status,
                    summary: String::new(),
                    timestamp_ms: 0,
                    pool_ordinal: job.pool_ordinal?,
                })
            })
            .collect());
    }
    let children =
        project_child_session_summaries_from_file(&child_sessions_dir(sessions_dir), &parent_path)?;
    Ok(SubagentPool::child_sessions_from_summaries(children))
}

fn project_child_session_view_with_children(
    sessions_dir: &Path,
    parent_session_id: String,
    children: Vec<ChildSessionSummary>,
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
    live_text: &ChildLiveText,
) -> Result<Option<ChildViewProjection>> {
    let Some(index) = select_child_navigation_index(&children, navigation, anchor_child_session_id)
    else {
        return Ok(None);
    };
    let child = &children[index];
    let records =
        read_child_session_records_allow_partial_tail(sessions_dir, &child.child_session_id)?;
    let snapshot = project_runtime_restore_snapshot(
        child.child_session_id.clone(),
        records,
        default_resume_cursor(),
        &[],
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    let records = snapshot.records;
    Ok(Some(ChildViewProjection {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        pool_ordinal: child.pool_ordinal,
        records,
        runtime_context,
        in_progress_assistant_text: live_text.get(&child.child_session_id),
    }))
}

#[cfg(test)]
#[path = "../../tests/internal/session/child_view/integration_tests.rs"]
mod integration_tests;
