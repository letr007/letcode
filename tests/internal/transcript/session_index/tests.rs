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
