use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{Result, anyhow};

use crate::transcript::{
    TranscriptRecorder, any_record_where, record_is_session_content, remove_empty_session_file,
};

pub(super) fn empty_session_path(path: &Path) -> Option<PathBuf> {
    (!any_record_where(path, record_is_session_content).ok()?).then(|| path.to_path_buf())
}

pub(super) fn remove_current_empty_session(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<bool> {
    let path = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();

    remove_empty_session_file(path)
}
