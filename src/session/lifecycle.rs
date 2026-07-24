//! Session lifecycle helpers shared by TUI and line CLI.
//!
//! Phase I covers **new-session transcript bootstrap** (create + start record +
//! swap). Context-scope prepare/apply and rich restore projection remain with
//! the frontend until SessionEngine owns them end-to-end.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

use crate::transcript::{TranscriptRecorder, remove_empty_session_file};

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
}
