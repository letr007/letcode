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
fn replacement_failures_preserve_unique_sources() {
    let dir = tempfile::tempdir().unwrap();
    let archive = ensure_archive_dir(dir.path()).unwrap();
    let path = archive.join(INDEX_FILE);
    fs::create_dir(&path).unwrap();
    let mut index = ArchiveIndexFile::default();
    let mut failed_sources = BTreeMap::new();

    for record_count in [3, 5] {
        index.sessions.insert("parent".into(), row(record_count));
        save_index(dir.path(), &index).unwrap_err();
        let sources: BTreeMap<_, _> = fs::read_dir(&archive)
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
    assert_eq!(rows(dir.path())["parent"].record_count, 5);
    for (source, bytes) in &failed_sources {
        assert_eq!(&fs::read(source).unwrap(), bytes);
    }
    assert_eq!(
        fs::read_dir(&archive)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|source| source.extension().and_then(|ext| ext.to_str()) == Some("tmp"))
            .count(),
        failed_sources.len()
    );
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
