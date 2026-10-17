use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 4;

/// Consecutive failures spent on one unchanged frontier. A later frontier starts a new series.
const MAX_SOURCE_FAILURES: u32 = 8;
const SOURCE_RETRY_DELAYS_MS: [u64; 5] = [
    60_000,
    5 * 60_000,
    30 * 60_000,
    2 * 60 * 60_000,
    6 * 60 * 60_000,
];

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
    pub failure_count: u32,
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

/// 巩固层对已有记忆的改动，只允许把冗余项合并进幸存者，或改写幸存者的正文。
/// 退休必须指名覆盖者（`into`），不允许无覆盖地删除。
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct CurationUpdate {
    #[serde(default)]
    pub merges: Vec<CurationMerge>,
    #[serde(default)]
    pub rewrites: Vec<CurationRewrite>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CurationMerge {
    pub retire: String,
    pub into: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CurationRewrite {
    pub id: String,
    pub title: String,
    pub summary: String,
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
    /// 被 memory__recall 返回过的次数与最近一次时间；读取路径记账，供巩固层判断哪些记忆真被用到。
    #[serde(default)]
    pub recall_count: u64,
    #[serde(default)]
    pub last_recalled_at: Option<u64>,
    /// 最近一次被巩固层改写的时刻；未被改写过则为 None。
    #[serde(default)]
    pub curated_at_ms: Option<u64>,
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
                    last_error TEXT,
                    failure_count INTEGER NOT NULL DEFAULT 0
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
                    state TEXT NOT NULL,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled_at INTEGER,
                    curated_at_ms INTEGER
                );
                CREATE INDEX memories_active_created
                    ON memories(state, created_at_ms DESC);",
            )?;
        } else {
            if version < 2 {
                transaction.execute_batch(
                    "ALTER TABLE sources
                        ADD COLUMN observed_sequence INTEGER NOT NULL DEFAULT 0;
                     UPDATE sources SET observed_sequence=cursor;",
                )?;
            }
            if version < 3 {
                transaction.execute_batch(
                    "ALTER TABLE memories ADD COLUMN recall_count INTEGER NOT NULL DEFAULT 0;
                     ALTER TABLE memories ADD COLUMN last_recalled_at INTEGER;
                     ALTER TABLE memories ADD COLUMN curated_at_ms INTEGER;",
                )?;
            }
            if version < 4 {
                transaction.execute_batch(
                    "ALTER TABLE sources ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0;",
                )?;
            }
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
                observed_sequence=MAX(sources.observed_sequence, excluded.observed_sequence),
                failure_count=CASE
                    WHEN excluded.observed_sequence > sources.observed_sequence THEN 0
                    ELSE sources.failure_count
                END,
                retry_at_ms=CASE
                    WHEN excluded.observed_sequence > sources.observed_sequence THEN 0
                    ELSE sources.retry_at_ms
                END,
                last_error=CASE
                    WHEN excluded.observed_sequence > sources.observed_sequence THEN NULL
                    ELSE sources.last_error
                END",
            params![session_id, path.to_string_lossy(), start_sequence],
        )?;
        Ok(())
    }

    pub(crate) fn sources(&self) -> Result<Vec<Source>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT session_id, path, cursor, observed_sequence, failure_count FROM sources
             WHERE cursor < observed_sequence
               AND failure_count < ?1
               AND retry_at_ms <= ?2
             ORDER BY failure_count, retry_at_ms, session_id",
        )?;
        Ok(statement
            .query_map(params![MAX_SOURCE_FAILURES, now_ms()], |row| {
                Ok(Source {
                    session_id: row.get(0)?,
                    path: PathBuf::from(row.get::<_, String>(1)?),
                    cursor: row.get(2)?,
                    observed_sequence: row.get(3)?,
                    failure_count: row.get(4)?,
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
                    retry_at_ms=0, last_error=NULL, failure_count=0
             WHERE session_id=?1 AND cursor=?3",
            params![source.session_id, end_sequence, source.cursor],
        )?;
        ensure!(changed == 1, "memory source changed before commit");
        transaction.commit()?;
        Ok(())
    }

    pub(crate) fn fail_source(&self, source: &Source, error: &str) -> Result<()> {
        let error = error.chars().take(400).collect::<String>();
        let delay = SOURCE_RETRY_DELAYS_MS[usize::try_from(source.failure_count)
            .unwrap_or(SOURCE_RETRY_DELAYS_MS.len() - 1)
            .min(SOURCE_RETRY_DELAYS_MS.len() - 1)];
        self.connect()?.execute(
            "UPDATE sources
             SET failure_count=failure_count+1, retry_at_ms=?2, last_error=?3
             WHERE session_id=?1 AND cursor=?4 AND observed_sequence=?5",
            params![
                source.session_id,
                now_ms().saturating_add(delay),
                error,
                source.cursor,
                source.observed_sequence
            ],
        )?;
        Ok(())
    }

    pub(crate) fn known_memories(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        self.read_active(limit.min(10_000))
    }

    /// 记录这些记忆刚被 memory__recall 返回过。失败只影响统计，不影响检索结果。
    pub(crate) fn record_recall(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut connection = self.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = now_ms();
        for id in ids {
            transaction.execute(
                "UPDATE memories SET recall_count=recall_count+1, last_recalled_at=?2
                 WHERE id=?1 AND state='active'",
                params![id, now],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }

    /// 应用一次巩固结果。整批在一个事务里，任一目标已不再是 active 就整体失败，
    /// 避免按已经过期的视图改动记忆。
    pub(crate) fn apply_curation(&self, update: &CurationUpdate) -> Result<()> {
        if update.merges.is_empty() && update.rewrites.is_empty() {
            return Ok(());
        }
        let mut connection = self.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = now_ms();
        for merge in &update.merges {
            let changed = transaction.execute(
                "UPDATE memories SET state='retired', curated_at_ms=?2
                 WHERE id=?1 AND state='active'",
                params![merge.retire, now],
            )?;
            ensure!(changed == 1, "memory curation target is no longer active");
        }
        for rewrite in &update.rewrites {
            let changed = transaction.execute(
                "UPDATE memories SET title=?2, summary=?3, curated_at_ms=?4
                 WHERE id=?1 AND state='active'",
                params![rewrite.id, rewrite.title, rewrite.summary, now],
            )?;
            ensure!(changed == 1, "memory curation target is no longer active");
        }
        transaction.commit()?;
        Ok(())
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
        let grams = query_grams(query.query.as_deref().unwrap_or_default());
        let candidates = candidates
            .into_iter()
            .map(|memory| {
                let fields = SearchFields::new(&memory);
                (memory, fields)
            })
            .collect::<Vec<_>>();
        let weights = gram_weights(&candidates, &grams);
        let mut ranked = candidates
            .into_iter()
            .filter_map(|(memory, fields)| {
                let score = memory_score(&fields, &grams, &weights);
                (grams.is_empty() || score > 0.0).then_some((score, memory))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_score, left), (right_score, right)| {
            right_score
                .total_cmp(left_score)
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
                    source_ids_json,paths_json,created_at_ms,state,
                    recall_count,last_recalled_at,curated_at_ms
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
                row.get::<_, u64>(11)?,
                row.get::<_, Option<u64>>(12)?,
                row.get::<_, Option<u64>>(13)?,
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
                recall_count,
                last_recalled_at,
                curated_at_ms,
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
                recall_count,
                last_recalled_at,
                curated_at_ms,
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

const MAX_QUERY_TERMS: usize = 24;

/// 连续汉字按二字组展开（`记忆系统` → `记忆`/`忆系`/`系统`），其余词保持原样。
/// 中文没有词边界，整句当成一个词时只剩逐字子串命中，长句必然查不到任何记忆。
fn query_grams(text: &str) -> Vec<String> {
    let mut grams = Vec::new();
    for term in text
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .take(MAX_QUERY_TERMS)
    {
        let term = term.to_lowercase();
        let mut run = String::new();
        let mut run_is_han = None;
        for character in term.chars() {
            let is_han = is_han(character);
            if run_is_han != Some(is_han) {
                push_grams(&mut grams, &run, run_is_han == Some(true));
                run.clear();
                run_is_han = Some(is_han);
            }
            run.push(character);
        }
        push_grams(&mut grams, &run, run_is_han == Some(true));
    }
    let mut seen = std::collections::HashSet::new();
    grams.retain(|gram| seen.insert(gram.clone()));
    grams
}

fn is_han(character: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&character)
}

fn push_grams(grams: &mut Vec<String>, run: &str, is_han: bool) {
    if run.is_empty() {
        return;
    }
    let characters = run.chars().collect::<Vec<_>>();
    if !is_han || characters.len() <= 2 {
        grams.push(run.to_string());
        return;
    }
    for pair in characters.windows(2) {
        grams.push(pair.iter().collect());
    }
}

/// 小写化后的可检索字段；文档频率与打分共用同一份，避免重复变换。
struct SearchFields {
    title: String,
    summary: String,
    paths: String,
}

impl SearchFields {
    fn new(memory: &MemoryRecord) -> Self {
        Self {
            title: memory.title.to_lowercase(),
            summary: memory.summary.to_lowercase(),
            paths: memory.paths.join(" ").to_lowercase(),
        }
    }

    fn contains(&self, gram: &str) -> bool {
        self.title.contains(gram) || self.summary.contains(gram) || self.paths.contains(gram)
    }
}

/// 文档频率取自本次候选集。中文长句会产生大量虚词二字组，不压低高频组时
/// 「命中很多常见组」会压过「命中稀有组」。
fn gram_weights(candidates: &[(MemoryRecord, SearchFields)], grams: &[String]) -> Vec<f64> {
    let total = candidates.len().max(1) as f64;
    grams
        .iter()
        .map(|gram| {
            let matching = candidates
                .iter()
                .filter(|(_, fields)| fields.contains(gram))
                .count() as f64;
            (1.0 + total / (1.0 + matching)).ln()
        })
        .collect()
}

fn memory_score(fields: &SearchFields, grams: &[String], weights: &[f64]) -> f64 {
    grams
        .iter()
        .zip(weights)
        .map(|(gram, weight)| {
            let hits = u64::from(fields.title.contains(gram)) * 6
                + u64::from(fields.summary.contains(gram)) * 2
                + u64::from(fields.paths.contains(gram)) * 8;
            weight * hits as f64
        })
        .sum()
}

#[cfg(test)]
#[path = "../../tests/internal/project_memory/store/tests.rs"]
mod tests;
