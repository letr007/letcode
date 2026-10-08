//! Sidecar index for fast `/resume` session listing.
//!
//! Stored as `sessions-index.json` next to `*.jsonl`. Entries are keyed by
//! session id and stamped with `(size, mtime_ms)` so stale rows are rebuilt
//! by rescanning only the mismatched transcript.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::SessionSummary;

const INDEX_FILE: &str = "sessions-index.json";
const INDEX_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionsIndexFile {
    version: u32,
    #[serde(default)]
    sessions: BTreeMap<String, IndexedSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct IndexedSession {
    size: u64,
    mtime_ms: u128,
    record_count: usize,
    first_timestamp_ms: Option<u128>,
    last_timestamp_ms: Option<u128>,
    model: Option<String>,
    title: Option<String>,
    last_user_summary: Option<String>,
    last_assistant_summary: Option<String>,
    workspace: Option<String>,
    has_content: bool,
}

impl IndexedSession {
    fn to_summary(&self, session_id: String) -> SessionSummary {
        SessionSummary {
            session_id,
            record_count: self.record_count,
            first_timestamp_ms: self.first_timestamp_ms,
            last_timestamp_ms: self.last_timestamp_ms,
            model: self.model.clone(),
            title: self.title.clone(),
            last_user_summary: self.last_user_summary.clone(),
            last_assistant_summary: self.last_assistant_summary.clone(),
            workspace: self.workspace.clone(),
        }
    }

    fn matches_stamp(&self, size: u64, mtime_ms: u128) -> bool {
        self.size == size && self.mtime_ms == mtime_ms
    }
}

fn index_path(base_dir: &Path) -> PathBuf {
    base_dir.join(INDEX_FILE)
}

fn file_stamp(path: &Path) -> Result<(u64, u128)> {
    let meta = fs::metadata(path)
        .with_context(|| format!("failed to stat transcript {}", path.display()))?;
    let mtime_ms = meta
        .modified()
        .unwrap_or(UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    Ok((meta.len(), mtime_ms))
}

fn load_index(base_dir: &Path) -> SessionsIndexFile {
    let path = index_path(base_dir);
    let Ok(bytes) = fs::read(&path) else {
        return SessionsIndexFile {
            version: INDEX_VERSION,
            sessions: BTreeMap::new(),
        };
    };
    match serde_json::from_slice::<SessionsIndexFile>(&bytes) {
        Ok(index) if index.version == INDEX_VERSION => index,
        _ => SessionsIndexFile {
            version: INDEX_VERSION,
            sessions: BTreeMap::new(),
        },
    }
}

fn save_index(base_dir: &Path, index: &SessionsIndexFile) -> Result<()> {
    fs::create_dir_all(base_dir)?;
    let path = index_path(base_dir);
    let bytes = serde_json::to_vec_pretty(index)?;
    let prefix = format!(".{INDEX_FILE}.");
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix).suffix(".tmp");
    #[cfg(unix)]
    builder.permissions(fs::Permissions::from_mode(0o666));
    let mut temp = builder.tempfile_in(base_dir).with_context(|| {
        format!(
            "failed to create session index temporary file in {}",
            base_dir.display()
        )
    })?;
    temp.write_all(&bytes)
        .with_context(|| format!("failed to write session index {}", temp.path().display()))?;
    temp.as_file()
        .sync_all()
        .with_context(|| format!("failed to sync session index {}", temp.path().display()))?;
    let tmp = temp.into_temp_path().keep()?;
    crate::config::replace_file(&tmp, &path)
        .with_context(|| format!("failed to replace session index {}", path.display()))?;
    Ok(())
}

pub(super) fn remove_session(base_dir: &Path, session_id: &str) {
    let mut index = load_index(base_dir);
    if index.sessions.remove(session_id).is_some() {
        let _ = save_index(base_dir, &index);
    }
}

/// List sessions using the sidecar index, rescanning only stale or missing rows.
pub(super) fn list_sessions_with_index(
    base_dir: &Path,
    mut summarize: impl FnMut(&Path, String) -> Result<Option<SessionSummary>>,
) -> Result<Vec<SessionSummary>> {
    let mut index = load_index(base_dir);
    let mut dirty = false;
    let mut live_ids = Vec::new();
    let mut sessions = Vec::new();

    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }
        let session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(session_id) => session_id.to_string(),
            None => continue,
        };
        live_ids.push(session_id.clone());

        let (size, mtime_ms) = file_stamp(&path)?;
        if let Some(cached) = index.sessions.get(&session_id)
            && cached.matches_stamp(size, mtime_ms)
        {
            if cached.has_content {
                sessions.push(cached.to_summary(session_id));
            }
            continue;
        }

        match summarize(&path, session_id.clone())? {
            Some(summary) => {
                let scanned_stamp = (size, mtime_ms);
                let current_stamp = file_stamp(&path)?;
                if current_stamp != scanned_stamp {
                    index.sessions.remove(&session_id);
                    dirty = true;
                } else {
                    let indexed = IndexedSession {
                        size,
                        mtime_ms,
                        record_count: summary.record_count,
                        first_timestamp_ms: summary.first_timestamp_ms,
                        last_timestamp_ms: summary.last_timestamp_ms,
                        model: summary.model.clone(),
                        title: summary.title.clone(),
                        last_user_summary: summary.last_user_summary.clone(),
                        last_assistant_summary: summary.last_assistant_summary.clone(),
                        workspace: summary.workspace.clone(),
                        has_content: true,
                    };
                    index.sessions.insert(session_id, indexed);
                    dirty = true;
                }
                sessions.push(summary);
            }
            None => {
                let scanned_stamp = (size, mtime_ms);
                let current_stamp = file_stamp(&path)?;
                if current_stamp != scanned_stamp {
                    index.sessions.remove(&session_id);
                } else {
                    index.sessions.insert(
                        session_id,
                        IndexedSession {
                            size,
                            mtime_ms,
                            has_content: false,
                            ..IndexedSession::default()
                        },
                    );
                }
                dirty = true;
            }
        }
    }

    let live: std::collections::HashSet<&str> = live_ids.iter().map(String::as_str).collect();
    let before = index.sessions.len();
    index.sessions.retain(|id, _| live.contains(id.as_str()));
    if index.sessions.len() != before {
        dirty = true;
    }

    if dirty {
        let _ = save_index(base_dir, &index);
    }

    sessions.sort_by_key(|session| session.last_timestamp_ms.unwrap_or(0));
    sessions.reverse();
    Ok(sessions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_failures_preserve_unique_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = index_path(dir.path());
        fs::create_dir(&path).unwrap();
        let mut index = SessionsIndexFile {
            version: INDEX_VERSION,
            ..SessionsIndexFile::default()
        };
        let mut failed_sources = BTreeMap::new();

        for record_count in [3, 5] {
            index.sessions.insert(
                "parent".into(),
                IndexedSession {
                    record_count,
                    ..IndexedSession::default()
                },
            );
            save_index(dir.path(), &index).unwrap_err();
            let sources: BTreeMap<_, _> = fs::read_dir(dir.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|source| source.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
                .map(|source| {
                    let bytes = fs::read(&source).unwrap();
                    (source, bytes)
                })
                .collect();
            assert_eq!(sources.len(), failed_sources.len() + 1);
            for (source, bytes) in &failed_sources {
                assert_eq!(sources.get(source), Some(bytes));
            }
            let expected = serde_json::to_vec_pretty(&index).unwrap();
            assert!(sources.values().any(|bytes| bytes == &expected));
            failed_sources = sources;
        }

        fs::remove_dir(&path).unwrap();
        save_index(dir.path(), &index).unwrap();
        assert_eq!(load_index(dir.path()).sessions["parent"].record_count, 5);
        for (source, bytes) in &failed_sources {
            assert_eq!(&fs::read(source).unwrap(), bytes);
        }
        assert_eq!(
            fs::read_dir(dir.path())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|source| source.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
                .count(),
            failed_sources.len()
        );
    }
}
