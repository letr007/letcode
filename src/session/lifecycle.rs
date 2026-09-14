//! Session lifecycle helpers shared by TUI and line CLI.
//!
//! Phase I covers **new-session transcript bootstrap** (create + start record +
//! swap). Phase K adds **resume resolve/open** helpers (prefix match + load +
//! open existing). Context-scope prepare/apply and rich restore projection
//! remain with the frontend until SessionEngine owns them end-to-end.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::agent::{Agent, PreparedPrimaryRoute};
use crate::runtime_context::{RuntimeActiveContext, RuntimeSnapshot};
use crate::session::archive::merged_session_summaries;
use crate::session::context_scope::{apply_prepared_context_scope, prepare_context_scope};
use crate::session::restore::project_runtime_restore_snapshot_with_children;
use crate::transcript::transcript_projection::{RuntimeRestoreSnapshot, SessionContextCursor};
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptFileFingerprint, TranscriptRecord, TranscriptRecorder,
    read_records, remove_empty_session_file, resolve_session_id,
};

/// Create a new on-disk session transcript and record the session-started event.
pub fn bootstrap_new_transcript(
    sessions_dir: impl AsRef<Path>,
    model: impl Into<String>,
) -> Result<TranscriptRecorder> {
    let mut recorder = TranscriptRecorder::create(sessions_dir.as_ref())?;
    if let Err(error) = recorder.record_session_started(model.into()) {
        let _ = remove_empty_session_file(recorder.path());
        return Err(error);
    }
    Ok(recorder)
}

/// Replace the live transcript recorder. Returns the previous transcript path
/// (callers may delete it when empty).
pub fn replace_live_transcript(
    live: &Arc<Mutex<TranscriptRecorder>>,
    new_recorder: TranscriptRecorder,
) -> Result<PathBuf> {
    let mut guard = live
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    let old_path = guard.path().to_path_buf();
    *guard = new_recorder;
    Ok(old_path)
}

/// Bootstrap a new session transcript and install it as the live recorder.
///
/// Does **not** reset the agent or re-apply context scope — callers do that
/// with their prepare/apply helpers, then optionally remove the old empty file.
#[cfg(test)]
pub fn start_new_transcript_session(
    live: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: impl AsRef<Path>,
    model: impl Into<String>,
) -> Result<PathBuf> {
    let new_recorder = bootstrap_new_transcript(sessions_dir, model)?;
    replace_live_transcript(live, new_recorder)
}

/// Best-effort removal of an empty previous session file after a successful swap.
pub fn cleanup_empty_session_file(path: PathBuf) -> Result<bool> {
    remove_empty_session_file(path)
}

/// Prepared new-session package for frontends that emit `SessionStarted` and
/// install via restore-snapshot.
pub struct PreparedNewSession {
    pub session_id: String,
    pub recorder: TranscriptRecorder,
    pub snapshot: RuntimeRestoreSnapshot,
    pub runtime_context: RuntimeActiveContext,
}

/// Bootstrap a new transcript and project its empty/root restore snapshot.
///
/// Does not mutate the agent or live recorder. On failure after bootstrap, the
/// new empty transcript file is removed.
pub fn prepare_new_session_package(
    sessions_dir: impl AsRef<Path>,
    model: impl Into<String>,
) -> Result<PreparedNewSession> {
    let sessions_dir = sessions_dir.as_ref();
    let mut recorder = bootstrap_new_transcript(sessions_dir, model)?;
    recorder.set_current_context_branch_id(None);
    let session_id = recorder.session_id().to_string();
    let new_path = recorder.path().to_path_buf();

    let prepare_result = (|| -> Result<PreparedNewSession> {
        let records = read_records(&new_path)?;
        let snapshot = project_runtime_restore_snapshot_with_children(
            session_id.clone(),
            records,
            SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: None,
            },
            sessions_dir,
        )?;
        let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
        Ok(PreparedNewSession {
            session_id: session_id.clone(),
            recorder,
            snapshot,
            runtime_context,
        })
    })();

    match prepare_result {
        Ok(prepared) => Ok(prepared),
        Err(error) => {
            let _ = remove_empty_session_file(&new_path);
            Err(error)
        }
    }
}

/// Install a prepared new session and clean a prior empty file.
/// All fallible agent preparation completes before the recorder swap.
/// All fallible validation required for a new-session switch, completed before
/// the caller cancels the outgoing session's subagents.
pub struct PreparedNewSessionInstall {
    pub(crate) prepared: PreparedNewSession,
    prepared_scope: crate::session::context_scope::PreparedContextScope,
    runtime_snapshot: RuntimeSnapshot,
    prepared_route: Option<PreparedPrimaryRoute>,
    old_path: PathBuf,
    new_path: PathBuf,
}

/// Prepare a new-session install while retaining a route prepared for commit.
pub(crate) fn prepare_new_session_install_with_route(
    agent: &Agent,
    live: &Arc<Mutex<TranscriptRecorder>>,
    prepared: PreparedNewSession,
    prepared_route: Option<PreparedPrimaryRoute>,
) -> Result<PreparedNewSessionInstall> {
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    let runtime_snapshot =
        agent.validate_runtime_snapshot_restore(prepared.snapshot.snapshot.clone())?;
    agent.prepare_new_session_permission_reset()?;
    let old_path = live
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();
    let new_path = prepared.recorder.path().to_path_buf();
    Ok(PreparedNewSessionInstall {
        prepared,
        prepared_scope,
        runtime_snapshot,
        prepared_route,
        old_path,
        new_path,
    })
}

impl PreparedNewSessionInstall {
    pub(crate) fn session(&self) -> &PreparedNewSession {
        &self.prepared
    }

    pub(crate) fn new_path(&self) -> &Path {
        &self.new_path
    }

    /// Commit a successfully prepared new-session install.
    ///
    /// This contains no business-fallible operations. A poisoned lock or an
    /// invalid validated snapshot is an internal invariant violation.
    pub(crate) fn commit(self, agent: &mut Agent, live: &Arc<Mutex<TranscriptRecorder>>) {
        let Self {
            prepared,
            prepared_scope,
            runtime_snapshot,
            prepared_route,
            old_path,
            new_path,
        } = self;
        let old_path_at_commit = replace_live_transcript(live, prepared.recorder)
            .expect("live transcript lock must remain healthy after preparation");
        debug_assert_eq!(old_path_at_commit, old_path);
        agent.install_new_session_runtime_snapshot(runtime_snapshot, prepared.snapshot.max_turn_id);
        apply_prepared_context_scope(agent, prepared_scope);
        if let Some(prepared_route) = prepared_route {
            agent.apply_prepared_route(prepared_route);
        }
        let _ = cleanup_replaced_empty_session(old_path, &new_path);
    }
}

/// Failure modes for resolving a session id from a user-supplied prefix.
#[derive(Debug)]
pub enum ResolveSessionError {
    EmptyQuery,
    ListFailed(anyhow::Error),
    NotFound { query: String },
    Ambiguous { query: String, matches: Vec<String> },
}

impl fmt::Display for ResolveSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuery => write!(f, "usage: /resume <session_id>"),
            Self::ListFailed(error) => write!(f, "failed to list sessions: {error}"),
            Self::NotFound { query } => write!(f, "session not found: {query}"),
            Self::Ambiguous { query, matches } => {
                write!(f, "multiple sessions match {query}: {}", matches.join(", "))
            }
        }
    }
}

impl std::error::Error for ResolveSessionError {}

/// Resolve a unique session id under `sessions_dir` from a prefix query.
///
/// Archived sessions are candidates like live ones, so a selection made from a
/// listing that includes archives resolves to the id the resume path expects.
pub fn resolve_session_prefix(
    sessions_dir: impl AsRef<Path>,
    query: &str,
) -> std::result::Result<String, ResolveSessionError> {
    if query.is_empty() {
        return Err(ResolveSessionError::EmptyQuery);
    }
    let sessions_dir = sessions_dir.as_ref();
    // Exact id (picker selection) must not pay for a full directory scan.
    if sessions_dir.join(format!("{query}.jsonl")).is_file() {
        return Ok(query.to_string());
    }
    let sessions =
        merged_session_summaries(sessions_dir).map_err(ResolveSessionError::ListFailed)?;
    match resolve_session_id(&sessions, query) {
        Ok(session_id) => Ok(session_id),
        Err(matches) if matches.is_empty() => Err(ResolveSessionError::NotFound {
            query: query.to_string(),
        }),
        Err(matches) => Err(ResolveSessionError::Ambiguous {
            query: query.to_string(),
            matches,
        }),
    }
}

/// Load transcript records for an existing session file.
#[cfg(test)]
pub fn load_session_records(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<Vec<TranscriptRecord>> {
    read_records(sessions_dir.as_ref().join(format!("{session_id}.jsonl")))
}

pub(crate) fn load_session_records_with_fingerprint(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<(Vec<TranscriptRecord>, TranscriptFileFingerprint)> {
    crate::transcript::read_resumable_records_with_fingerprint(
        sessions_dir.as_ref().join(format!("{session_id}.jsonl")),
    )
}

/// Open an existing session transcript for resume (append-safe open).
#[cfg(test)]
pub fn open_resume_transcript(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<TranscriptRecorder> {
    TranscriptRecorder::open_existing(sessions_dir.as_ref(), session_id)
}

/// Open an existing session transcript for resume using records already loaded
/// from that transcript.
#[cfg(test)]
pub fn open_resume_transcript_with_records(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
    records: &[TranscriptRecord],
) -> Result<TranscriptRecorder> {
    TranscriptRecorder::open_existing_with_records(sessions_dir.as_ref(), session_id, records)
}

pub(crate) fn open_resume_transcript_with_records_at_fingerprint(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
    records: &[TranscriptRecord],
    fingerprint: &TranscriptFileFingerprint,
) -> Result<TranscriptRecorder> {
    TranscriptRecorder::open_existing_with_records_at_fingerprint(
        sessions_dir.as_ref(),
        session_id,
        records,
        fingerprint,
    )
}

/// After a successful resume/new swap, remove the previous path when it is a
/// different empty session file.
pub fn cleanup_replaced_empty_session(old_path: PathBuf, new_path: &Path) -> Result<bool> {
    if old_path != new_path {
        cleanup_empty_session_file(old_path)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use crate::session::archive::{ArchiveBudget, SessionArchiveConfig, run_archive_pass};
    use crate::transcript::{archive_index, child_sessions_dir};

    const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

    fn write_session(base_dir: &Path, model: &str) -> String {
        let mut recorder = TranscriptRecorder::create(base_dir).unwrap();
        let session_id = recorder.session_id().to_string();
        recorder.record_session_started(model).unwrap();
        recorder.record_user_message("question").unwrap();
        recorder.record_assistant_message("answer").unwrap();
        session_id
    }

    /// A parent transcript with one child session, both aged past the archive
    /// threshold so the next pass compresses the whole family.
    fn write_aged_family(sessions_dir: &Path) -> (String, String) {
        let child_id = write_session(&child_sessions_dir(sessions_dir), "child-model");
        let parent_id = write_session(sessions_dir, "parent-model");
        let mut recorder = TranscriptRecorder::open_existing(sessions_dir, &parent_id).unwrap();
        recorder
            .record_subagent_started(
                "run-1", &parent_id, "run-1", &child_id, "explorer", "started", 1,
            )
            .unwrap();
        drop(recorder);

        let modified = SystemTime::now() - Duration::from_secs(8 * SECONDS_PER_DAY);
        for path in [
            sessions_dir.join(format!("{parent_id}.jsonl")),
            child_sessions_dir(sessions_dir).join(format!("{child_id}.jsonl")),
        ] {
            let file = fs::OpenOptions::new().write(true).open(path).unwrap();
            file.set_times(fs::FileTimes::new().set_modified(modified))
                .unwrap();
        }
        (parent_id, child_id)
    }

    fn archive_family(sessions_dir: &Path) -> String {
        let (parent_id, _) = write_aged_family(sessions_dir);
        let report = run_archive_pass(
            sessions_dir,
            SessionArchiveConfig::default(),
            ArchiveBudget::default(),
        )
        .unwrap();
        assert_eq!(report.archived_sessions, vec![parent_id.clone()]);
        assert!(!sessions_dir.join(format!("{parent_id}.jsonl")).exists());
        parent_id
    }

    #[test]
    fn an_archived_family_resolves_by_id_and_prefix_then_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let parent_id = archive_family(dir.path());

        assert_eq!(
            resolve_session_prefix(dir.path(), &parent_id).unwrap(),
            parent_id
        );
        let prefix = &parent_id[..8];
        assert_eq!(
            resolve_session_prefix(dir.path(), prefix).unwrap(),
            parent_id
        );

        let prepared = crate::session::prepare_resume_package(dir.path(), &parent_id).unwrap();

        assert_eq!(prepared.session_id, parent_id);
        assert!(dir.path().join(format!("{parent_id}.jsonl")).is_file());
        assert!(archive_index::archived_ids(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn an_unknown_query_still_reports_not_found_next_to_archived_sessions() {
        let dir = tempfile::tempdir().unwrap();
        archive_family(dir.path());

        let error = resolve_session_prefix(dir.path(), "no-such-session").unwrap_err();

        assert!(
            matches!(error, ResolveSessionError::NotFound { .. }),
            "expected NotFound, got {error:?}"
        );
    }

    #[test]
    fn a_prefix_matching_two_archived_sessions_is_ambiguous() {
        let dir = tempfile::tempdir().unwrap();
        let parent_id = archive_family(dir.path());
        // The same transcript under a neighbouring id: an exact id that is also
        // a prefix of another archived session must stay ambiguous.
        let sibling_id = format!("{parent_id}-sibling");
        let archived = fs::read(archive_index::transcript_path(dir.path(), &parent_id)).unwrap();
        fs::write(
            archive_index::transcript_path(dir.path(), &sibling_id),
            archived,
        )
        .unwrap();

        let error = resolve_session_prefix(dir.path(), &parent_id).unwrap_err();
        let matches = match error {
            ResolveSessionError::Ambiguous { matches, .. } => matches,
            other => panic!("expected Ambiguous, got {other}"),
        };

        let mut matches = matches;
        matches.sort();
        assert_eq!(matches, vec![parent_id, sibling_id]);
    }
}
