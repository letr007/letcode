use super::*;

fn match_item(path: &str, line: u64, text: &str) -> Value {
    json!({"path": path, "line": line, "column": 1, "text": text})
}

#[tokio::test]
async fn large_result_set_folds_to_artifact_with_preview() {
    let matches = (0..200)
        .map(|i| match_item("src/a.rs", i as u64 + 1, &"x".repeat(300)))
        .collect::<Vec<_>>();
    let map = present_search_matches(matches, false).await.unwrap();
    assert_eq!(map.get("folded").and_then(Value::as_bool), Some(true));
    assert_eq!(map.get("truncated").and_then(Value::as_bool), Some(true));
    assert_eq!(map.get("total_matches").and_then(Value::as_u64), Some(200));
    assert_eq!(map.get("files").and_then(Value::as_u64), Some(1));
    let preview = map.get("matches").and_then(Value::as_array).unwrap();
    assert_eq!(preview.len(), super::SEARCH_PREVIEW_MATCHES);
    assert!(
        preview
            .iter()
            .all(|m| m.get("text_truncated") == Some(&json!(true))),
        "long lines in preview are truncated"
    );
    let path = map.get("local_path").and_then(Value::as_str).unwrap();
    let on_disk = std::fs::read(path).unwrap();
    assert_eq!(on_disk.split(|&b| b == b'\n').count(), 200);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn small_result_set_stays_inline() {
    let matches = vec![
        match_item("src/a.rs", 1, "hello"),
        match_item("src/b.rs", 2, "world"),
    ];
    let map = present_search_matches(matches, false).await.unwrap();
    assert_eq!(map.get("folded"), None);
    assert_eq!(map.get("local_path"), None);
    assert_eq!(map.get("truncated").and_then(Value::as_bool), Some(false));
    assert_eq!(map.get("files").and_then(Value::as_u64), Some(2));
    let matches = map.get("matches").and_then(Value::as_array).unwrap();
    assert_eq!(matches.len(), 2);
    assert!(matches.iter().all(|m| m.get("text_truncated").is_none()));
}
