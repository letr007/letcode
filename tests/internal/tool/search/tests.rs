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

#[test]
fn context_events_keep_order_and_omit_column() {
    let stdout = [
        r#"{"type":"begin","data":{"path":{"text":"src/a.rs"}}}"#,
        r#"{"type":"context","data":{"path":{"text":"src/a.rs"},"lines":{"text":"before\n"},"line_number":9,"submatches":[]}}"#,
        r#"{"type":"match","data":{"path":{"text":"src/a.rs"},"lines":{"text":"hit\n"},"line_number":10,"submatches":[{"start":4}]}}"#,
        r#"{"type":"context","data":{"path":{"text":"src/a.rs"},"lines":{"text":"after\n"},"line_number":11,"submatches":[]}}"#,
        r#"{"type":"end","data":{"path":{"text":"src/a.rs"}}}"#,
    ]
    .join("\n");

    let (items, truncated) = parse_match_events(&stdout, 100);

    assert!(!truncated);
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].get("is_context"), Some(&json!(true)));
    assert_eq!(items[0].get("column"), None);
    assert_eq!(items[1].get("is_context"), None);
    assert_eq!(items[1].get("column"), Some(&json!(5)));
    assert_eq!(items[2].get("is_context"), Some(&json!(true)));
}

#[test]
fn context_rows_do_not_consume_max_results() {
    let stdout = [
        r#"{"type":"context","data":{"path":{"text":"a"},"lines":{"text":"c\n"},"line_number":1,"submatches":[]}}"#,
        r#"{"type":"match","data":{"path":{"text":"a"},"lines":{"text":"m\n"},"line_number":2,"submatches":[{"start":0}]}}"#,
    ]
    .join("\n");

    let (items, truncated) = parse_match_events(&stdout, 1);

    assert!(!truncated);
    assert_eq!(items.len(), 2);
}

#[test]
fn file_paths_split_on_null() {
    let paths = parse_file_paths("src/a.rs\u{0}src/b.rs\u{0}");

    assert_eq!(paths, vec!["src/a.rs", "src/b.rs"]);
}

#[test]
fn counts_split_path_from_count() {
    let (counts, total) = parse_counts("src/a.rs\u{0}2\nsrc/b.rs\u{0}5\n");

    assert_eq!(counts.len(), 2);
    assert_eq!(
        counts[0].get("path").and_then(Value::as_str),
        Some("src/a.rs")
    );
    assert_eq!(counts[0].get("count").and_then(Value::as_u64), Some(2));
    assert_eq!(
        counts[1].get("path").and_then(Value::as_str),
        Some("src/b.rs")
    );
    assert_eq!(counts[1].get("count").and_then(Value::as_u64), Some(5));
    assert_eq!(total, 7);
}
