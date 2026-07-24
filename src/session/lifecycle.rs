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
use async_openai::config::Config;

use crate::agent::Agent;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::context_scope::{apply_prepared_context_scope, prepare_context_scope};
use crate::session::restore::project_runtime_restore_snapshot_with_children;
use crate::transcript::transcript_projection::{RuntimeRestoreSnapshot, SessionContextCursor};
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptRecord, TranscriptRecorder, list_sessions, read_records,
    remove_empty_session_file, resolve_session_id,
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
/// install via restore-snapshot (TUI). CLI may use the simpler
/// [`install_new_session_for_agent`] path instead.
pub struct PreparedNewSession {
    pub session_id: String,
    pub recorder: TranscriptRecorder,
    pub snapshot: RuntimeRestoreSnapshot,
    pub runtime_context: RuntimeActiveContext,
}

/// Bootstrap a new transcript and project its empty/root restore snapshot.
///
/// On failure after bootstrap, the new empty transcript file is removed.
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

/// Apply a prepared new-session package onto the agent (restore empty runtime +
/// context-scope). Does **not** swap the live recorder.
pub fn apply_prepared_new_session_to_agent<C: Config>(
    agent: &mut Agent<C>,
    prepared: &PreparedNewSession,
) -> Result<()> {
    agent.restore_new_session_runtime_snapshot(
        prepared.snapshot.protocol_frames.clone(),
        prepared.snapshot.snapshot.clone(),
        prepared.snapshot.max_turn_id,
    )?;
    let prepared_scope = prepare_context_scope(&prepared.recorder)?;
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(())
}

/// CLI-style new session install: bootstrap transcript, reset agent, apply
/// context-scope, swap live recorder, and clean the previous empty file.
///
/// Prefer this for line CLI. TUI uses [`prepare_new_session_package`] so it can
/// emit `SessionStarted` with restore projection payloads.
pub fn install_new_session_for_agent<C: Config>(
    agent: &mut Agent<C>,
    live: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: impl AsRef<Path>,
) -> Result<String> {
    let new_recorder = bootstrap_new_transcript(sessions_dir, agent.model().to_string())?;
    let prepared_scope = prepare_context_scope(&new_recorder)?;
    let session_id = new_recorder.session_id().to_string();
    agent.reset_for_new_session();
    apply_prepared_context_scope(agent, prepared_scope);
    let old_path = replace_live_transcript(live, new_recorder)?;
    let _ = cleanup_empty_session_file(old_path);
    Ok(session_id)
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
pub fn resolve_session_prefix(
    sessions_dir: impl AsRef<Path>,
    query: &str,
) -> std::result::Result<String, ResolveSessionError> {
    if query.is_empty() {
        return Err(ResolveSessionError::EmptyQuery);
    }
    let sessions =
        list_sessions(sessions_dir.as_ref()).map_err(ResolveSessionError::ListFailed)?;
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
pub fn load_session_records(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<Vec<TranscriptRecord>> {
    read_records(sessions_dir.as_ref().join(format!("{session_id}.jsonl")))
}

/// Open an existing session transcript for resume (append-safe open).
pub fn open_resume_transcript(
    sessions_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<TranscriptRecorder> {
    TranscriptRecorder::open_existing(sessions_dir.as_ref(), session_id)
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn bootstrap_and_replace_swaps_session_id() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-lifecycle-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let first = TranscriptRecorder::create(&base_dir).expect("first");
        let first_id = first.session_id().to_string();
        let live = Arc::new(Mutex::new(first));

        let old_path =
            start_new_transcript_session(&live, &base_dir, "gpt-test").expect("start new");
        let second_id = live.lock().expect("lock").session_id().to_string();
        assert_ne!(first_id, second_id);
        assert!(old_path.ends_with(format!("{first_id}.jsonl")));
        let _ = cleanup_empty_session_file(old_path);
    }

    #[test]
    fn resolve_session_prefix_finds_unique_and_reports_ambiguous() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-resolve-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut a = bootstrap_new_transcript(&base_dir, "m").expect("a");
        a.record_user_message("alpha").expect("user a");
        let a_id = a.session_id().to_string();
        let mut b = bootstrap_new_transcript(&base_dir, "m").expect("b");
        b.record_user_message("beta").expect("user b");
        let b_id = b.session_id().to_string();
        drop(a);
        drop(b);

        assert_eq!(
            resolve_session_prefix(&base_dir, &a_id).expect("exact"),
            a_id
        );
        assert!(matches!(
            resolve_session_prefix(&base_dir, ""),
            Err(ResolveSessionError::EmptyQuery)
        ));
        assert!(matches!(
            resolve_session_prefix(&base_dir, "does-not-exist-zzzz"),
            Err(ResolveSessionError::NotFound { .. })
        ));

        // Shared timestamp/pid prefixes can collide; if both share a common
        // prefix, expect Ambiguous rather than a silent pick.
        let common: String = a_id
            .chars()
            .zip(b_id.chars())
            .take_while(|(l, r)| l == r)
            .map(|(l, _)| l)
            .collect();
        if !common.is_empty() && a_id != b_id {
            match resolve_session_prefix(&base_dir, &common) {
                Ok(id) => assert!(id == a_id || id == b_id),
                Err(ResolveSessionError::Ambiguous { matches, .. }) => {
                    assert!(matches.len() >= 2);
                }
                Err(other) => panic!("unexpected resolve error: {other}"),
            }
        }

        let records = load_session_records(&base_dir, &a_id).expect("records");
        assert!(!records.is_empty());
        let opened = open_resume_transcript(&base_dir, &a_id).expect("open");
        assert_eq!(opened.session_id(), a_id);
    }
}
