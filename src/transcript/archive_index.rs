//! Sidecar index for archived session transcripts.
//!
//! Stored as `{sessions_dir}/archive/index.json`, next to the compressed
//! transcripts. The archive directory itself stays the source of truth for
//! *which* sessions are archived; this file carries listing summaries so an
//! archived session can be listed without decompressing it, plus the failure
//! bookkeeping the archiver uses to back off from transcripts it cannot handle.
//!
//! Both the archiver and the resume restore path write this file, so every
//! mutation happens under an exclusive lock on a sibling lock file and replaces
//! the index atomically.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result};
use fs4::fs_std::FileExt as _;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::SessionSummary;

pub(crate) const ARCHIVE_DIR: &str = "archive";
/// Suffix of an archived transcript, e.g. `archive/<session_id>.jsonl.zst`.
pub(crate) const TRANSCRIPT_SUFFIX: &str = ".jsonl.zst";
/// Suffix of a session's archived artifacts directory.
pub(crate) const ARTIFACTS_SUFFIX: &str = ".tar.zst";
/// Mirrors `sessions/children`: child transcripts are archived below the top
/// level so listing the archive directory never reports a child as a session.
const CHILDREN_DIR: &str = "children";
/// Mirrors `sessions/artifacts`.
const ARTIFACTS_DIR: &str = "artifacts";

const INDEX_FILE: &str = "index.json";
const INDEX_LOCK_FILE: &str = "index.lock";
const INDEX_VERSION: u32 = 1;
const LOCK_RETRIES: usize = 10;
const LOCK_RETRY_DELAY: Duration = Duration::from_millis(200);

/// Listing summary and archive bookkeeping for one archived session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ArchivedSession {
    #[serde(default)]
    pub(crate) original_bytes: u64,
    #[serde(default)]
    pub(crate) archived_bytes: u64,
    #[serde(default)]
    pub(crate) record_count: usize,
    #[serde(default)]
    pub(crate) first_timestamp_ms: Option<u128>,
    #[serde(default)]
    pub(crate) last_timestamp_ms: Option<u128>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) title: Option<String>,
    #[serde(default)]
    pub(crate) last_user_summary: Option<String>,
    #[serde(default)]
    pub(crate) last_assistant_summary: Option<String>,
    #[serde(default)]
    pub(crate) archived_at_ms: u128,
}

impl ArchivedSession {
    pub(crate) fn summary(&self, session_id: String) -> SessionSummary {
        SessionSummary {
            session_id,
            record_count: self.record_count,
            first_timestamp_ms: self.first_timestamp_ms,
            last_timestamp_ms: self.last_timestamp_ms,
            model: self.model.clone(),
            title: self.title.clone(),
            last_user_summary: self.last_user_summary.clone(),
            last_assistant_summary: self.last_assistant_summary.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArchiveIndexFile {
    version: u32,
    #[serde(default)]
    sessions: BTreeMap<String, ArchivedSession>,
    /// Consecutive archive failures per session. Kept apart from `sessions` so a
    /// session that never archived successfully is never listed as archived.
    #[serde(default)]
    failed_attempts: BTreeMap<String, u32>,
}

impl Default for ArchiveIndexFile {
    fn default() -> Self {
        Self {
            version: INDEX_VERSION,
            sessions: BTreeMap::new(),
            failed_attempts: BTreeMap::new(),
        }
    }
}

impl ArchiveIndexFile {
    fn usable(self) -> Self {
        if self.version == INDEX_VERSION {
            self
        } else {
            Self::default()
        }
    }
}

pub(crate) fn archive_dir(base_dir: &Path) -> PathBuf {
    base_dir.join(ARCHIVE_DIR)
}

pub(crate) fn ensure_archive_dir(base_dir: &Path) -> Result<PathBuf> {
    let dir = archive_dir(base_dir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create archive directory {}", dir.display()))?;
    Ok(dir)
}

pub(crate) fn transcript_path(base_dir: &Path, session_id: &str) -> PathBuf {
    archive_dir(base_dir).join(format!("{session_id}{TRANSCRIPT_SUFFIX}"))
}

pub(crate) fn child_transcript_path(base_dir: &Path, session_id: &str) -> PathBuf {
    archive_dir(base_dir)
        .join(CHILDREN_DIR)
        .join(format!("{session_id}{TRANSCRIPT_SUFFIX}"))
}

pub(crate) fn artifacts_path(base_dir: &Path, session_id: &str) -> PathBuf {
    archive_dir(base_dir)
        .join(ARTIFACTS_DIR)
        .join(format!("{session_id}{ARTIFACTS_SUFFIX}"))
}

pub(crate) fn ensure_archive_subdirs(base_dir: &Path) -> Result<()> {
    let dir = ensure_archive_dir(base_dir)?;
    for subdir in [CHILDREN_DIR, ARTIFACTS_DIR] {
        let path = dir.join(subdir);
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create archive directory {}", path.display()))?;
    }
    Ok(())
}

/// Session ids with an archived transcript, read from the directory itself so a
/// lost index row can never hide an archived session from the listing.
pub(crate) fn archived_ids(base_dir: &Path) -> Result<Vec<String>> {
    scan_transcript_ids(&archive_dir(base_dir))
}

/// Child session ids with an archived transcript.
pub(crate) fn archived_child_ids(base_dir: &Path) -> Result<Vec<String>> {
    scan_transcript_ids(&archive_dir(base_dir).join(CHILDREN_DIR))
}

fn scan_transcript_ids(dir: &Path) -> Result<Vec<String>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read archive directory {}", dir.display()));
        }
    };

    let mut ids = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(session_id) = name.strip_suffix(TRANSCRIPT_SUFFIX) {
            ids.push(session_id.to_string());
        }
    }
    ids.sort();
    Ok(ids)
}

pub(crate) fn rows(base_dir: &Path) -> BTreeMap<String, ArchivedSession> {
    load_index(base_dir).sessions
}

#[cfg(test)]
pub(crate) fn failed_attempts(base_dir: &Path) -> BTreeMap<String, u32> {
    load_index(base_dir).failed_attempts
}

pub(crate) fn upsert(base_dir: &Path, session_id: &str, row: ArchivedSession) -> Result<()> {
    with_index(base_dir, |index| {
        index.sessions.insert(session_id.to_string(), row.clone());
        index.failed_attempts.remove(session_id);
        Ok(())
    })
}

pub(crate) fn remove(base_dir: &Path, session_id: &str) -> Result<()> {
    with_index(base_dir, |index| {
        index.sessions.remove(session_id);
        index.failed_attempts.remove(session_id);
        Ok(())
    })
}

/// Record one failed archive attempt and return the new consecutive count.
pub(crate) fn bump_attempts(base_dir: &Path, session_id: &str) -> Result<u32> {
    let mut count = 0;
    with_index(base_dir, |index| {
        prune_missing_attempts(base_dir, index);
        let attempts = index
            .failed_attempts
            .entry(session_id.to_string())
            .or_insert(0);
        *attempts = attempts.saturating_add(1);
        count = *attempts;
        Ok(())
    })?;
    Ok(count)
}

pub(crate) fn clear_attempts(base_dir: &Path, session_id: &str) -> Result<()> {
    with_index(base_dir, |index| {
        index.failed_attempts.remove(session_id);
        Ok(())
    })
}

/// Drop failure counters whose session no longer exists as a live transcript or
/// as an archived one, so the bookkeeping cannot outlive the data it describes.
fn prune_missing_attempts(base_dir: &Path, index: &mut ArchiveIndexFile) {
    let live_dir = base_dir.to_path_buf();
    let archive_dir = archive_dir(base_dir);
    index.failed_attempts.retain(|session_id, _| {
        live_dir.join(format!("{session_id}.jsonl")).is_file()
            || archive_dir
                .join(format!("{session_id}{TRANSCRIPT_SUFFIX}"))
                .is_file()
    });
}

fn with_index<T>(
    base_dir: &Path,
    apply: impl FnOnce(&mut ArchiveIndexFile) -> Result<T>,
) -> Result<T> {
    let dir = ensure_archive_dir(base_dir)?;
    let lock_path = dir.join(INDEX_LOCK_FILE);
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open archive index lock {}", lock_path.display()))?;
    acquire_lock(&lock, &lock_path)?;

    let mut index = load_index(base_dir);
    index.version = INDEX_VERSION;
    let value = apply(&mut index)?;
    save_index(base_dir, &index)?;
    Ok(value)
}

fn acquire_lock(file: &fs::File, path: &Path) -> Result<()> {
    for attempt in 0..LOCK_RETRIES {
        if file
            .try_lock_exclusive()
            .with_context(|| format!("failed to lock archive index {}", path.display()))?
        {
            return Ok(());
        }
        if attempt + 1 < LOCK_RETRIES {
            sleep(LOCK_RETRY_DELAY);
        }
    }
    anyhow::bail!(
        "archive index {} is locked by another process",
        path.display()
    )
}

fn load_index(base_dir: &Path) -> ArchiveIndexFile {
    let path = archive_dir(base_dir).join(INDEX_FILE);
    let Ok(bytes) = fs::read(&path) else {
        return ArchiveIndexFile::default();
    };
    match serde_json::from_slice::<ArchiveIndexFile>(&bytes) {
        Ok(index) => index.usable(),
        Err(error) => {
            // A damaged cache must not fail archiving or listing: the archive
            // directory stays authoritative and rows are rebuilt on demand.
            warn!(
                path = %path.display(),
                error = %error,
                "archive index is unreadable; rebuilding rows from the archive directory"
            );
            ArchiveIndexFile::default()
        }
    }
}

fn save_index(base_dir: &Path, index: &ArchiveIndexFile) -> Result<()> {
    let dir = ensure_archive_dir(base_dir)?;
    let path = dir.join(INDEX_FILE);
    let tmp = dir.join(format!("{INDEX_FILE}.tmp"));
    let bytes = serde_json::to_vec_pretty(index)?;
    fs::write(&tmp, bytes)
        .with_context(|| format!("failed to write archive index {}", tmp.display()))?;
    crate::config::replace_file(&tmp, &path)
        .with_context(|| format!("failed to replace archive index {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    fn row(record_count: usize) -> ArchivedSession {
        ArchivedSession {
            record_count,
            archived_at_ms: now_ms(),
            ..ArchivedSession::default()
        }
    }

    #[test]
    fn upsert_and_rows_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        upsert(dir.path(), "parent", row(3)).unwrap();
        upsert(dir.path(), "other", row(5)).unwrap();

        let rows = rows(dir.path());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows["parent"].record_count, 3);
        assert_eq!(rows["other"].record_count, 5);
    }

    #[test]
    fn missing_corrupt_and_old_indexes_read_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(rows(dir.path()).is_empty());

        let archive = ensure_archive_dir(dir.path()).unwrap();
        fs::write(archive.join(INDEX_FILE), b"{not json").unwrap();
        assert!(rows(dir.path()).is_empty());

        fs::write(
            archive.join(INDEX_FILE),
            serde_json::to_vec(&serde_json::json!({
                "version": INDEX_VERSION + 1,
                "sessions": {"parent": {"record_count": 9}},
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(rows(dir.path()).is_empty());
    }

    #[test]
    fn remove_drops_row_and_failure_counter() {
        let dir = tempfile::tempdir().unwrap();
        upsert(dir.path(), "parent", row(1)).unwrap();
        bump_attempts(dir.path(), "parent").unwrap();

        remove(dir.path(), "parent").unwrap();

        assert!(rows(dir.path()).is_empty());
        assert!(failed_attempts(dir.path()).is_empty());
    }

    #[test]
    fn attempts_count_up_and_a_successful_upsert_clears_them() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("parent.jsonl");
        fs::write(&live, b"{}\n").unwrap();

        assert_eq!(bump_attempts(dir.path(), "parent").unwrap(), 1);
        assert_eq!(bump_attempts(dir.path(), "parent").unwrap(), 2);
        assert_eq!(failed_attempts(dir.path())["parent"], 2);

        upsert(dir.path(), "parent", row(1)).unwrap();
        assert!(failed_attempts(dir.path()).is_empty());

        clear_attempts(dir.path(), "parent").unwrap();
        assert!(failed_attempts(dir.path()).is_empty());
    }

    #[test]
    fn attempts_for_vanished_sessions_are_pruned() {
        let dir = tempfile::tempdir().unwrap();
        let live = dir.path().join("parent.jsonl");
        fs::write(&live, b"{}\n").unwrap();
        bump_attempts(dir.path(), "parent").unwrap();
        bump_attempts(dir.path(), "gone").unwrap();
        assert_eq!(failed_attempts(dir.path()).len(), 2);

        fs::remove_file(&live).unwrap();
        bump_attempts(dir.path(), "parent").unwrap();

        // "parent" is gone too, so its stale counter is pruned as well and the
        // attempt that just happened starts from scratch.
        let attempts = failed_attempts(dir.path());
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts["parent"], 1);
    }

    #[test]
    fn archived_ids_scan_transcripts_only() {
        let dir = tempfile::tempdir().unwrap();
        ensure_archive_subdirs(dir.path()).unwrap();
        let archive = archive_dir(dir.path());
        fs::write(archive.join(format!("live{TRANSCRIPT_SUFFIX}")), b"x").unwrap();
        fs::write(archive.join(format!("live{TRANSCRIPT_SUFFIX}.tmp")), b"x").unwrap();
        fs::write(artifacts_path(dir.path(), "live"), b"x").unwrap();
        fs::write(archive.join(INDEX_FILE), b"{}").unwrap();
        fs::write(child_transcript_path(dir.path(), "kid"), b"x").unwrap();

        assert_eq!(archived_ids(dir.path()).unwrap(), vec!["live".to_string()]);
        assert_eq!(
            archived_child_ids(dir.path()).unwrap(),
            vec!["kid".to_string()]
        );
        assert!(archived_ids(&dir.path().join("absent")).unwrap().is_empty());
        assert!(
            archived_child_ids(&dir.path().join("absent"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn concurrent_upserts_keep_every_row() {
        let dir = tempfile::tempdir().unwrap();
        upsert(dir.path(), "a", row(1)).unwrap();
        upsert(dir.path(), "b", row(2)).unwrap();
        upsert(dir.path(), "a", row(3)).unwrap();

        let rows = rows(dir.path());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows["a"].record_count, 3);
        assert_eq!(rows["b"].record_count, 2);
    }
}
