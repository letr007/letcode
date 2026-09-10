use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 2;

#[derive(Clone)]
pub(crate) struct MemoryStore {
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub session_id: String,
    pub path: PathBuf,
    pub cursor: u64,
    pub observed_sequence: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MemoryDraft {
    pub kind: String,
    pub title: String,
    pub summary: String,
    #[serde(default = "default_status")]
    pub status: String,
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub supersedes: Vec<String>,
}

fn default_status() -> String {
    "useful".into()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct MemoryUpdate {
    #[serde(default)]
    pub memories: Vec<MemoryDraft>,
    #[serde(default)]
    pub withdrawn_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct MemoryRecord {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub summary: String,
    pub status: String,
    pub session_id: String,
    pub branch_id: String,
    pub source_ids: Vec<String>,
    pub paths: Vec<String>,
    pub created_at_ms: u64,
    pub state: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct StoreStatus {
    pub state: String,
    pub active_memories: usize,
    pub tracked_sources: usize,
    pub pending_sources: usize,
    pub last_error: Option<String>,
}

impl MemoryStore {
    pub(crate) fn open(root: &Path, workspace: &Path) -> Result<Self> {
        let workspace = workspace
            .canonicalize()
            .context("failed to resolve project memory workspace")?;
        let project_key = format!(
            "{:x}",
            Sha256::digest(workspace.as_os_str().as_encoded_bytes())
        );
        let directory = root.join(project_key);
        std::fs::create_dir_all(&directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
        }
        let store = Self {
            path: directory.join("memory.sqlite"),
        };
        store.initialize()?;
        Ok(store)
    }

    fn initialize(&self) -> Result<()> {
        let mut connection = self.connect()?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let version: u32 =
            transaction.pragma_query_value(None, "user_version", |row| row.get(0))?;
        ensure!(
            version <= SCHEMA_VERSION,
            "unsupported project memory schema {version}"
        );
        if version == 0 {
            transaction.execute_batch(
                "CREATE TABLE sources (
                    session_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    cursor INTEGER NOT NULL,
                    observed_sequence INTEGER NOT NULL,
                    retry_at_ms INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                );
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    paths_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    state TEXT NOT NULL
                );
                CREATE INDEX memories_active_created
                    ON memories(state, created_at_ms DESC);",
            )?;
        } else if version == 1 {
            transaction.execute_batch(
                "ALTER TABLE sources
                    ADD COLUMN observed_sequence INTEGER NOT NULL DEFAULT 0;
                 UPDATE sources SET observed_sequence=cursor;",
            )?;
        }
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        transaction.commit()?;
        Ok(())
    }

    fn connect(&self) -> Result<Connection> {
        let connection = Connection::open(&self.path)?;
        connection.busy_timeout(Duration::from_secs(2))?;
        Ok(connection)
    }

    pub(crate) fn worker_lock_path(&self) -> PathBuf {
        self.path.with_extension("lock")
    }

    pub(crate) fn register_source(
        &self,
        session_id: &str,
        path: &Path,
        start_sequence: u64,
    ) -> Result<()> {
        ensure!(
            !session_id.trim().is_empty(),
            "memory source session is empty"
        );
        let path = path
            .canonicalize()
            .context("failed to resolve memory source journal")?;
        self.connect()?.execute(
            "INSERT INTO sources(session_id, path, cursor, observed_sequence)
             VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT(session_id) DO UPDATE SET
                path=excluded.path,
                observed_sequence=MAX(sources.observed_sequence, excluded.observed_sequence)",
            params![session_id, path.to_string_lossy(), start_sequence],
        )?;
        Ok(())
    }

    pub(crate) fn sources(&self) -> Result<Vec<Source>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT session_id, path, cursor, observed_sequence FROM sources
             WHERE cursor < observed_sequence AND retry_at_ms <= ?1
             ORDER BY retry_at_ms, session_id",
        )?;
        Ok(statement
            .query_map([now_ms()], |row| {
                Ok(Source {
                    session_id: row.get(0)?,
                    path: PathBuf::from(row.get::<_, String>(1)?),
                    cursor: row.get(2)?,
                    observed_sequence: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub(crate) fn complete(
        &self,
        source: &Source,
        end_sequence: u64,
        branch_id: &str,
        update: &MemoryUpdate,
        known_ids: &[String],
    ) -> Result<()> {
        ensure!(
            end_sequence > source.cursor && end_sequence <= source.observed_sequence,
            "memory source cursor did not advance within its observed frontier"
        );
        ensure!(
            update.memories.len() <= 20 && update.withdrawn_ids.len() <= 20,
            "memory update is too large"
        );
        let mut connection = self.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let actual_cursor: u64 = transaction.query_row(
            "SELECT cursor FROM sources WHERE session_id=?1",
            [&source.session_id],
            |row| row.get(0),
        )?;
        ensure!(
            actual_cursor == source.cursor,
            "memory source advanced concurrently"
        );
        for draft in &update.memories {
            validate_draft(draft)?;
        }
        let mut revised = std::collections::BTreeSet::new();
        for target in update
            .withdrawn_ids
            .iter()
            .chain(update.memories.iter().flat_map(|memory| &memory.supersedes))
        {
            ensure!(
                known_ids.contains(target),
                "memory revision references unseen memory"
            );
            ensure!(revised.insert(target), "memory is revised more than once");
            let changed = transaction.execute(
                "UPDATE memories SET state='retired' WHERE id=?1 AND state='active'",
                [target],
            )?;
            ensure!(changed == 1, "memory revision target is not active");
        }
        let created_at_ms = now_ms();
        for (ordinal, draft) in update.memories.iter().enumerate() {
            let id = format!("memory:{}:{end_sequence}:{ordinal}", source.session_id);
            transaction.execute(
                "INSERT INTO memories(
                    id, kind, title, summary, status, session_id, branch_id,
                    source_ids_json, paths_json, created_at_ms, state
                ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,'active')",
                params![
                    id,
                    draft.kind,
                    draft.title,
                    draft.summary,
                    draft.status,
                    source.session_id,
                    branch_id,
                    serde_json::to_string(&draft.source_ids)?,
                    serde_json::to_string(&draft.paths)?,
                    created_at_ms,
                ],
            )?;
        }
        let changed = transaction.execute(
            "UPDATE sources SET cursor=?2,
                    observed_sequence=MAX(observed_sequence, ?2),
                    retry_at_ms=0, last_error=NULL
             WHERE session_id=?1 AND cursor=?3",
            params![source.session_id, end_sequence, source.cursor],
        )?;
        ensure!(changed == 1, "memory source changed before commit");
        transaction.commit()?;
        Ok(())
    }

    pub(crate) fn fail_source(&self, source: &Source, error: &str) -> Result<()> {
        let error = error.chars().take(400).collect::<String>();
        self.connect()?.execute(
            "UPDATE sources SET retry_at_ms=?2, last_error=?3
             WHERE session_id=?1 AND cursor=?4",
            params![source.session_id, now_ms() + 60_000, error, source.cursor],
        )?;
        Ok(())
    }

    pub(crate) fn known_memories(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        self.read_active(limit.min(10_000))
    }

    pub(crate) fn query(
        &self,
        query: &crate::memory::MemoryRecallQuery,
    ) -> Result<Vec<MemoryRecord>> {
        let mut candidates = self.read_active(10_000)?;
        candidates.retain(|memory| {
            (query.kinds.is_empty()
                || query
                    .kinds
                    .iter()
                    .any(|kind| kind_name(*kind) == memory.kind))
                && (query.statuses.is_empty()
                    || query
                        .statuses
                        .iter()
                        .any(|status| status_name(*status) == memory.status))
                && (query.paths.is_empty()
                    || query.paths.iter().any(|path| {
                        memory.paths.iter().any(|candidate| {
                            candidate == path
                                || candidate.starts_with(&format!("{path}/"))
                                || path.starts_with(&format!("{candidate}/"))
                        })
                    }))
        });
        let terms = query
            .query
            .as_deref()
            .unwrap_or_default()
            .split_whitespace()
            .filter(|term| !term.is_empty())
            .take(24)
            .map(|term| term.to_lowercase())
            .collect::<Vec<_>>();
        let mut ranked = candidates
            .into_iter()
            .filter_map(|memory| {
                let score = memory_score(&memory, &terms);
                (terms.is_empty() || score > 0).then_some((score, memory))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .cmp(left_score)
                .then_with(|| right.created_at_ms.cmp(&left.created_at_ms))
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(ranked
            .into_iter()
            .take(query.limit.min(20))
            .map(|(_, memory)| memory)
            .collect())
    }

    pub(crate) fn status(&self) -> Result<StoreStatus> {
        let connection = self.connect()?;
        let active_memories = connection.query_row(
            "SELECT COUNT(*) FROM memories WHERE state='active'",
            [],
            |row| row.get(0),
        )?;
        let tracked_sources =
            connection.query_row("SELECT COUNT(*) FROM sources", [], |row| row.get(0))?;
        let pending_sources = connection.query_row(
            "SELECT COUNT(*) FROM sources WHERE cursor < observed_sequence",
            [],
            |row| row.get(0),
        )?;
        let last_error = connection
            .query_row(
                "SELECT last_error FROM sources WHERE last_error IS NOT NULL
                 ORDER BY retry_at_ms DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let state = if last_error.is_some() {
            "failed"
        } else if tracked_sources == 0 {
            "missing"
        } else if pending_sources > 0 {
            "synchronizing"
        } else if active_memories == 0 {
            "empty"
        } else {
            "ready"
        };
        Ok(StoreStatus {
            state: state.into(),
            active_memories,
            tracked_sources,
            pending_sources,
            last_error,
        })
    }

    fn read_active(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT id,kind,title,summary,status,session_id,branch_id,
                    source_ids_json,paths_json,created_at_ms,state
             FROM memories WHERE state='active'
             ORDER BY created_at_ms DESC,id LIMIT ?1",
        )?;
        let rows = statement.query_map([limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, u64>(9)?,
                row.get::<_, String>(10)?,
            ))
        })?;
        rows.map(|row| {
            let (
                id,
                kind,
                title,
                summary,
                status,
                session_id,
                branch_id,
                source_ids_json,
                paths_json,
                created_at_ms,
                state,
            ) = row?;
            Ok(MemoryRecord {
                id,
                kind,
                title,
                summary,
                status,
                session_id,
                branch_id,
                source_ids: serde_json::from_str(&source_ids_json)?,
                paths: serde_json::from_str(&paths_json)?,
                created_at_ms,
                state,
            })
        })
        .collect()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

pub(crate) fn validate_draft(draft: &MemoryDraft) -> Result<()> {
    ensure!(
        ["decision", "validation", "diagnostic", "experiment_result"]
            .contains(&draft.kind.as_str()),
        "invalid memory kind"
    );
    ensure!(
        ["useful", "active", "blocked", "dead_end"].contains(&draft.status.as_str()),
        "invalid memory status"
    );
    ensure!(
        !draft.title.trim().is_empty()
            && draft.title.chars().count() <= 512
            && !draft.summary.trim().is_empty()
            && draft.summary.chars().count() <= 8192,
        "invalid memory text"
    );
    ensure!(
        !draft.source_ids.is_empty()
            && draft.source_ids.len() <= 128
            && draft.paths.len() <= 20
            && draft.paths.iter().all(|path| path.chars().count() <= 1024),
        "invalid memory sources or paths"
    );
    Ok(())
}

fn kind_name(kind: crate::memory::MemoryKind) -> &'static str {
    match kind {
        crate::memory::MemoryKind::ExperimentResult => "experiment_result",
        crate::memory::MemoryKind::Decision => "decision",
        crate::memory::MemoryKind::Validation => "validation",
        crate::memory::MemoryKind::Diagnostic => "diagnostic",
    }
}

fn status_name(status: crate::memory::MemoryStatus) -> &'static str {
    match status {
        crate::memory::MemoryStatus::Active => "active",
        crate::memory::MemoryStatus::Useful => "useful",
        crate::memory::MemoryStatus::DeadEnd => "dead_end",
        crate::memory::MemoryStatus::Blocked => "blocked",
    }
}

fn memory_score(memory: &MemoryRecord, terms: &[String]) -> u64 {
    let title = memory.title.to_lowercase();
    let summary = memory.summary.to_lowercase();
    let paths = memory.paths.join(" ").to_lowercase();
    terms
        .iter()
        .map(|term| {
            u64::from(title.contains(term)) * 6
                + u64::from(summary.contains(term)) * 2
                + u64::from(paths.contains(term)) * 8
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft() -> MemoryDraft {
        MemoryDraft {
            kind: "decision".into(),
            title: "缓存设计".into(),
            summary: "Use stable cache keys".into(),
            status: "useful".into(),
            source_ids: vec!["raw:2".into()],
            paths: vec!["src/cache.rs".into()],
            supersedes: vec![],
        }
    }

    fn query(text: &str) -> crate::memory::MemoryRecallQuery {
        crate::memory::validate_memory_recall_query(&serde_json::json!({"query":text})).unwrap()
    }

    #[test]
    fn draft_text_limits_count_unicode_characters() {
        let mut memory = draft();
        memory.title = "记".repeat(512);
        memory.summary = "忆".repeat(8192);
        memory.paths = vec!["路".repeat(1024)];
        validate_draft(&memory).unwrap();

        memory.title.push('超');
        assert!(validate_draft(&memory).is_err());
    }

    #[test]
    fn project_scope_restart_and_text_search_do_not_need_the_source_journal() {
        let root = tempfile::tempdir().unwrap();
        let project_a = tempfile::tempdir().unwrap();
        let project_b = tempfile::tempdir().unwrap();
        let journal = tempfile::NamedTempFile::new().unwrap();
        let store = MemoryStore::open(root.path(), project_a.path()).unwrap();
        store.register_source("s", journal.path(), 1).unwrap();
        store.register_source("s", journal.path(), 3).unwrap();
        let source = store.sources().unwrap().remove(0);
        assert_eq!(source.cursor, 1);
        assert_eq!(source.observed_sequence, 3);
        assert_eq!(store.status().unwrap().state, "synchronizing");
        store
            .complete(
                &source,
                3,
                "main",
                &MemoryUpdate {
                    memories: vec![draft()],
                    withdrawn_ids: vec![],
                },
                &[],
            )
            .unwrap();
        drop(journal);
        let store = MemoryStore::open(root.path(), project_a.path()).unwrap();
        assert_eq!(store.status().unwrap().state, "ready");
        for text in ["缓存", "cache other", "src/cache.rs", "stable"] {
            assert_eq!(store.query(&query(text)).unwrap().len(), 1, "{text}");
        }
        assert!(store.query(&query("%' OR 1=1 --")).unwrap().is_empty());
        let other = MemoryStore::open(root.path(), project_b.path()).unwrap();
        assert!(other.known_memories(20).unwrap().is_empty());
        assert_eq!(other.status().unwrap().state, "missing");
    }

    #[test]
    fn opens_version_one_store_without_reimporting_sources() {
        let root = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let workspace = project.path().canonicalize().unwrap();
        let project_key = format!(
            "{:x}",
            Sha256::digest(workspace.as_os_str().as_encoded_bytes())
        );
        let directory = root.path().join(project_key);
        std::fs::create_dir_all(&directory).unwrap();
        let connection = Connection::open(directory.join("memory.sqlite")).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (
                    session_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    cursor INTEGER NOT NULL,
                    retry_at_ms INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                );
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    branch_id TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    paths_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    state TEXT NOT NULL
                );
                INSERT INTO sources(session_id,path,cursor)
                    VALUES ('old','/tmp/old.jsonl',42);
                PRAGMA user_version=1;",
            )
            .unwrap();
        drop(connection);

        let store = MemoryStore::open(root.path(), project.path()).unwrap();
        assert!(store.sources().unwrap().is_empty());
        let status = store.status().unwrap();
        assert_eq!(status.state, "empty");
        assert_eq!(status.pending_sources, 0);
    }

    #[test]
    fn cursor_and_revisions_commit_atomically() {
        let root = tempfile::tempdir().unwrap();
        let journal = tempfile::NamedTempFile::new().unwrap();
        let store = MemoryStore::open(root.path(), root.path()).unwrap();
        store.register_source("s", journal.path(), 0).unwrap();
        store.register_source("s", journal.path(), 2).unwrap();
        let original = store.sources().unwrap().remove(0);
        store
            .complete(
                &original,
                2,
                "main",
                &MemoryUpdate {
                    memories: vec![draft()],
                    withdrawn_ids: vec![],
                },
                &[],
            )
            .unwrap();
        assert!(
            store
                .complete(&original, 3, "main", &MemoryUpdate::default(), &[])
                .is_err()
        );
        store.register_source("s", journal.path(), 4).unwrap();
        let source = store.sources().unwrap().remove(0);
        let id = store.known_memories(1).unwrap()[0].id.clone();
        assert!(
            store
                .complete(
                    &source,
                    4,
                    "main",
                    &MemoryUpdate {
                        memories: vec![],
                        withdrawn_ids: vec![id.clone(), "unknown".into()],
                    },
                    std::slice::from_ref(&id),
                )
                .is_err()
        );
        assert_eq!(store.sources().unwrap()[0].cursor, 2);
        assert_eq!(store.known_memories(20).unwrap().len(), 1);
        store
            .complete(
                &source,
                4,
                "main",
                &MemoryUpdate {
                    memories: vec![],
                    withdrawn_ids: vec![id.clone()],
                },
                &[id],
            )
            .unwrap();
        store.register_source("s", journal.path(), 5).unwrap();
        let source = store.sources().unwrap().remove(0);
        store
            .complete(&source, 5, "main", &MemoryUpdate::default(), &[])
            .unwrap();
        assert!(store.known_memories(20).unwrap().is_empty());
        assert_eq!(store.status().unwrap().state, "empty");
        store.register_source("s", journal.path(), 6).unwrap();
        let source = store.sources().unwrap().remove(0);
        store.fail_source(&source, "test error").unwrap();
        assert!(store.sources().unwrap().is_empty());
        let status = store.status().unwrap();
        assert_eq!(status.state, "failed");
        assert_eq!(status.last_error.as_deref(), Some("test error"));
    }
}
