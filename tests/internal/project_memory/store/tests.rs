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

fn memory_with(title: &str, summary: &str, paths: &[&str]) -> MemoryDraft {
    MemoryDraft {
        kind: "diagnostic".into(),
        title: title.into(),
        summary: summary.into(),
        status: "useful".into(),
        source_ids: vec!["raw:2".into()],
        paths: paths.iter().map(|path| (*path).into()).collect(),
        supersedes: vec![],
    }
}

#[test]
fn opens_version_two_store_and_backfills_recall_columns() {
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
                    last_error TEXT,
                    observed_sequence INTEGER NOT NULL DEFAULT 0
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
                INSERT INTO sources(session_id,path,cursor,observed_sequence)
                    VALUES ('old','/tmp/old.jsonl',1,1);
                INSERT INTO memories(id,kind,title,summary,status,session_id,branch_id,
                    source_ids_json,paths_json,created_at_ms,state)
                    VALUES ('memory:old:1:0','decision','旧记忆','正文','useful','old','main',
                    '[\"raw:1\"]','[]',1,'active');
                PRAGMA user_version=2;",
        )
        .unwrap();
    drop(connection);

    let store = MemoryStore::open(root.path(), project.path()).unwrap();
    let memories = store.known_memories(10).unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].title, "旧记忆");
    assert_eq!(memories[0].recall_count, 0);
    assert_eq!(memories[0].last_recalled_at, None);
    assert_eq!(memories[0].curated_at_ms, None);
    assert_eq!(store.status().unwrap().state, "ready");
}

#[test]
fn recall_accounting_counts_each_returned_memory() {
    let root = tempfile::tempdir().unwrap();
    let journal = tempfile::NamedTempFile::new().unwrap();
    let store = MemoryStore::open(root.path(), root.path()).unwrap();
    store.register_source("s", journal.path(), 1).unwrap();
    store.register_source("s", journal.path(), 2).unwrap();
    let source = store.sources().unwrap().remove(0);
    store
        .complete(
            &source,
            2,
            "main",
            &MemoryUpdate {
                memories: vec![draft()],
                withdrawn_ids: vec![],
            },
            &[],
        )
        .unwrap();
    let hits = store.query(&query("缓存")).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].recall_count, 0);
    let ids = hits
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    store.record_recall(&ids).unwrap();
    let memories = store.known_memories(10).unwrap();
    assert_eq!(memories[0].recall_count, 1);
    assert!(memories[0].last_recalled_at.is_some());
    store.record_recall(&ids).unwrap();
    assert_eq!(store.known_memories(10).unwrap()[0].recall_count, 2);
}

#[test]
fn curation_retires_and_rewrites_active_memories() {
    let root = tempfile::tempdir().unwrap();
    let journal = tempfile::NamedTempFile::new().unwrap();
    let store = MemoryStore::open(root.path(), root.path()).unwrap();
    store.register_source("s", journal.path(), 1).unwrap();
    store.register_source("s", journal.path(), 2).unwrap();
    let source = store.sources().unwrap().remove(0);
    store
        .complete(
            &source,
            2,
            "main",
            &MemoryUpdate {
                memories: vec![
                    memory_with("缓存失效策略", "短正文", &[]),
                    memory_with("缓存失效策略与回源", "长正文", &[]),
                ],
                withdrawn_ids: vec![],
            },
            &[],
        )
        .unwrap();
    let active = store.known_memories(10).unwrap();
    assert_eq!(active.len(), 2);
    let survivor = active
        .iter()
        .find(|memory| memory.title == "缓存失效策略与回源")
        .unwrap();
    let duplicate = active
        .iter()
        .find(|memory| memory.title == "缓存失效策略")
        .unwrap();
    let update = CurationUpdate {
        merges: vec![CurationMerge {
            retire: duplicate.id.clone(),
            into: survivor.id.clone(),
        }],
        rewrites: vec![CurationRewrite {
            id: survivor.id.clone(),
            title: survivor.title.clone(),
            summary: "合并后的正文".into(),
        }],
    };
    store.apply_curation(&update).unwrap();
    let active = store.known_memories(10).unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].summary, "合并后的正文");
    assert!(active[0].curated_at_ms.is_some());

    // 同一批再应用一次必须整体失败，因为目标已经不是 active。
    assert!(store.apply_curation(&update).is_err());
    assert_eq!(store.known_memories(10).unwrap().len(), 1);
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
    assert!(store.query(&query("，。！？")).unwrap().is_empty());
    let other = MemoryStore::open(root.path(), project_b.path()).unwrap();
    assert!(other.known_memories(20).unwrap().is_empty());
    assert_eq!(other.status().unwrap().state, "missing");
}

#[test]
fn query_grams_expand_han_runs_and_keep_other_tokens() {
    assert_eq!(
        query_grams("记忆系统 store.rs"),
        ["记忆", "忆系", "系统", "store.rs"]
    );
    assert_eq!(query_grams("cache other"), ["cache", "other"]);
    assert_eq!(query_grams("记忆 记忆"), ["记忆"]);
    assert_eq!(
        query_grams("，stdio模式 spawn，"),
        ["stdio", "模式", "spawn"]
    );
    assert_eq!(
        query_grams("`src/a_b.rs` (foo::bar) /tmp/a.txt v1.2.3"),
        ["src/a_b.rs", "foo::bar", "/tmp/a.txt", "v1.2.3"]
    );
    assert_eq!(
        query_grams("!= foo! ? => foo::<T> @"),
        ["!=", "foo!", "?", "=>", "foo::<t>", "@"]
    );
    assert!(query_grams("，。！？").is_empty());
    assert!(query_grams(" \t\n").is_empty());
}

#[test]
fn chinese_sentences_recall_by_character_bigrams() {
    let root = tempfile::tempdir().unwrap();
    let journal = tempfile::NamedTempFile::new().unwrap();
    let store = MemoryStore::open(root.path(), root.path()).unwrap();
    store.register_source("s", journal.path(), 1).unwrap();
    store.register_source("s", journal.path(), 2).unwrap();
    let source = store.sources().unwrap().remove(0);
    store
        .complete(
            &source,
            2,
            "main",
            &MemoryUpdate {
                memories: vec![memory_with(
                    "记忆检索打分",
                    "中文问句按二字组命中",
                    &["src/project_memory/store.rs"],
                )],
                withdrawn_ids: vec![],
            },
            &[],
        )
        .unwrap();
    for text in ["记忆检索的打分是怎么做的", "打分是怎么算出来的"] {
        let hits = store.query(&query(text)).unwrap();
        assert_eq!(hits.len(), 1, "{text}");
        assert_eq!(hits[0].title, "记忆检索打分");
    }
}

#[test]
fn rare_grams_outrank_common_ones() {
    // 「检索」出现在 4/5 条候选，「打分」只在 1 条。不计文档频率时 paths(8) 命中会压过 title(6)。
    let root = tempfile::tempdir().unwrap();
    let journal = tempfile::NamedTempFile::new().unwrap();
    let store = MemoryStore::open(root.path(), root.path()).unwrap();
    store.register_source("common", journal.path(), 1).unwrap();
    store.register_source("common", journal.path(), 2).unwrap();
    let common = store.sources().unwrap().remove(0);
    store
        .complete(
            &common,
            2,
            "main",
            &MemoryUpdate {
                memories: (0..4)
                    .map(|ordinal| {
                        memory_with(&format!("缓存策略 {ordinal}"), "无关内容", &["资料/检索"])
                    })
                    .collect(),
                withdrawn_ids: vec![],
            },
            &[],
        )
        .unwrap();
    store.register_source("rare", journal.path(), 1).unwrap();
    store.register_source("rare", journal.path(), 2).unwrap();
    let rare = store.sources().unwrap().remove(0);
    store
        .complete(
            &rare,
            2,
            "main",
            &MemoryUpdate {
                memories: vec![memory_with("打分标准", "无关内容", &[])],
                withdrawn_ids: vec![],
            },
            &[],
        )
        .unwrap();
    let hits = store.query(&query("检索打分")).unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].title, "打分标准");
}

#[test]
fn query_scores_reward_coverage() {
    let focused = SearchFields {
        title: String::new(),
        summary: "alpha beta gamma".into(),
        paths: String::new(),
    };
    let repeated = SearchFields {
        title: "alpha".into(),
        summary: String::new(),
        paths: "alpha".into(),
    };
    let grams = query_grams("alpha beta gamma");
    let weights = [1.0; 3];
    assert!(memory_score(&focused, &grams, &weights) > memory_score(&repeated, &grams, &weights));
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

#[test]
fn repeated_failures_back_off_then_stop_until_the_frontier_moves() {
    let root = tempfile::tempdir().unwrap();
    let journal = tempfile::NamedTempFile::new().unwrap();
    let store = MemoryStore::open(root.path(), root.path()).unwrap();
    store.register_source("s", journal.path(), 0).unwrap();
    store.register_source("s", journal.path(), 2).unwrap();
    let mut previous_retry = 0u64;
    for _ in 0..MAX_SOURCE_FAILURES {
        let source = store.sources().unwrap().remove(0);
        let before = now_ms();
        store.fail_source(&source, "provider failed").unwrap();
        let (count, retry_at): (u32, u64) = store
            .connect()
            .unwrap()
            .query_row(
                "SELECT failure_count, retry_at_ms FROM sources WHERE session_id='s'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, source.failure_count + 1);
        assert!(retry_at >= before.saturating_add(60_000));
        assert!(retry_at > previous_retry);
        previous_retry = retry_at;
        store
            .connect()
            .unwrap()
            .execute("UPDATE sources SET retry_at_ms=0 WHERE session_id='s'", [])
            .unwrap();
    }
    assert!(store.sources().unwrap().is_empty());
    assert_eq!(store.status().unwrap().pending_sources, 1);

    store.register_source("s", journal.path(), 2).unwrap();
    assert!(store.sources().unwrap().is_empty());

    store.register_source("s", journal.path(), 3).unwrap();
    let row: (u64, u64, u32, u64, Option<String>) = store
        .connect()
        .unwrap()
        .query_row(
            "SELECT cursor, observed_sequence, failure_count, retry_at_ms, last_error
                 FROM sources WHERE session_id='s'",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        row,
        (0, 3, 0, 0, None),
        "frontier advance should reset retries"
    );
    let resumed = store.sources().unwrap().remove(0);
    assert_eq!(resumed.cursor, 0);
    assert_eq!(resumed.observed_sequence, 3);
    assert_eq!(resumed.failure_count, 0);
    assert_eq!(store.status().unwrap().state, "synchronizing");
}
