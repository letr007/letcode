//! Sidecar index for fast `/resume` session listing.
//!
//! Stored as `sessions-index.json` next to `*.jsonl`. Entries are keyed by
//! session id and stamped with `(size, mtime_ms)` so stale rows are rebuilt
//! by rescanning only the mismatched transcript.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::{SessionSummary, TranscriptRecord, TranscriptEvent, summarize_text};

const INDEX_FILE: &str = "sessions-index.json";
const INDEX_VERSION: u32 = 1;

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
        }
    }

    fn matches_stamp(&self, size: u64, mtime_ms: u128) -> bool {
        self.size == size && self.mtime_ms == mtime_ms
    }

    fn apply_record(&mut self, record: &TranscriptRecord) {
        self.record_count = self.record_count.saturating_add(1);
        if self.first_timestamp_ms.is_none() {
            self.first_timestamp_ms = Some(record.timestamp_ms);
        }
        self.last_timestamp_ms = Some(record.timestamp_ms);
        match &record.event {
            TranscriptEvent::SessionStarted { model } => self.model = Some(model.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => self.model = Some(new_model.clone()),
            TranscriptEvent::SessionTitle { title } => self.title = Some(title.clone()),
            TranscriptEvent::UserMessage { content } => {
                self.has_content = true;
                self.last_user_summary = Some(summarize_text(&content.display_text()));
            }
            TranscriptEvent::AssistantMessage { content } => {
                self.has_content = true;
                self.last_assistant_summary = Some(summarize_text(content));
            }
            event if event.is_session_content() => self.has_content = true,
            _ => {}
        }
    }

    fn refresh_stamp(&mut self, path: &Path) {
        if let Ok((size, mtime_ms)) = file_stamp(path) {
            self.size = size;
            self.mtime_ms = mtime_ms;
        }
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
    let tmp = base_dir.join(format!("{INDEX_FILE}.tmp"));
    let bytes = serde_json::to_vec_pretty(index)?;
    fs::write(&tmp, bytes)
        .with_context(|| format!("failed to write session index {}", tmp.display()))?;
    fs::rename(&tmp, &path)
        .with_context(|| format!("failed to replace session index {}", path.display()))?;
    Ok(())
}

fn affects_session_index(event: &TranscriptEvent) -> bool {
    matches!(
        event,
        TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::SessionTitle { .. }
            | TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::AssistantMessage { .. }
    ) || event.is_session_content()
}

/// Best-effort incremental update after a successful journal append.
pub(super) fn upsert_from_record(transcript_path: &Path, record: &TranscriptRecord) {
    if !affects_session_index(&record.event) {
        return;
    }
    let Some(base_dir) = transcript_path.parent() else {
        return;
    };
    let mut index = load_index(base_dir);
    let entry = index
        .sessions
        .entry(record.session_id.clone())
        .or_default();
    entry.apply_record(record);
    entry.refresh_stamp(transcript_path);
    let _ = save_index(base_dir, &index);
}

pub(super) fn upsert_from_records(transcript_path: &Path, records: &[TranscriptRecord]) {
    let relevant = records
        .iter()
        .filter(|record| affects_session_index(&record.event))
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return;
    }
    let Some(base_dir) = transcript_path.parent() else {
        return;
    };
    let mut index = load_index(base_dir);
    let session_id = relevant[0].session_id.clone();
    let entry = index.sessions.entry(session_id).or_default();
    for record in relevant {
        entry.apply_record(record);
    }
    entry.refresh_stamp(transcript_path);
    let _ = save_index(base_dir, &index);
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
    summarize: impl Fn(&Path, String) -> Result<Option<SessionSummary>>,
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
                let mut indexed = IndexedSession {
                    size,
                    mtime_ms,
                    record_count: summary.record_count,
                    first_timestamp_ms: summary.first_timestamp_ms,
                    last_timestamp_ms: summary.last_timestamp_ms,
                    model: summary.model.clone(),
                    title: summary.title.clone(),
                    last_user_summary: summary.last_user_summary.clone(),
                    last_assistant_summary: summary.last_assistant_summary.clone(),
                    has_content: true,
                };
                // Re-stat after scan in case the file moved under us.
                indexed.refresh_stamp(&path);
                index.sessions.insert(session_id, indexed);
                dirty = true;
                sessions.push(summary);
            }
            None => {
                index.sessions.insert(
                    session_id,
                    IndexedSession {
                        size,
                        mtime_ms,
                        has_content: false,
                        ..IndexedSession::default()
                    },
                );
                dirty = true;
            }
        }
    }

    let live: std::collections::HashSet<&str> =
        live_ids.iter().map(String::as_str).collect();
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
