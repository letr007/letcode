use super::*;
use std::path::Path;

#[test]
fn preview_keeps_only_leading_chars() {
    let content = "abcdefghij";
    assert_eq!(fold_preview(content, 4), "abcd");
    assert_eq!(fold_preview(content, 100), content);
    assert_eq!(fold_preview("中文内容", 2), "中文");
}

#[test]
fn artifact_names_are_content_addressed() {
    let name = artifact_file_name(b"hello", "txt");
    assert_eq!(name, artifact_file_name(b"hello", "txt"));
    assert_ne!(name, artifact_file_name(b"hello!", "txt"));
    assert!(name.ends_with(".txt"));
    assert_ne!(
        artifact_file_name(b"x", "out"),
        artifact_file_name(b"x", "err")
    );
}

#[test]
fn trusted_artifact_path_covers_only_our_temp_subdirs() {
    let root = std::env::temp_dir()
        .canonicalize()
        .unwrap_or_else(|_| std::env::temp_dir());
    let command = root.join(COMMAND_ARTIFACT_DIR).join("stream-1-2.out");
    let fetch = root.join(FETCH_ARTIFACT_DIR).join("a.txt");
    let search = root.join(SEARCH_ARTIFACT_DIR).join("deadbeef.txt");
    assert!(is_trusted_artifact_path(Path::new(&command)));
    assert!(is_trusted_artifact_path(Path::new(&fetch)));
    assert!(is_trusted_artifact_path(Path::new(&search)));
    // Sibling / unrelated paths and the temp root itself are not trusted.
    assert!(!is_trusted_artifact_path(Path::new(&root)));
    assert!(!is_trusted_artifact_path(Path::new(
        &root.join("letcode-outside-tool-read-123.txt")
    )));
    assert!(!is_trusted_artifact_path(Path::new(
        &root.join("letcode-command-other")
    )));
    assert!(!is_trusted_artifact_path(Path::new(
        "/nope/letcode-command/x"
    )));
}

#[tokio::test]
async fn write_artifact_is_idempotent_and_round_trips() {
    let body = b"payload content";
    let first = write_artifact("letcode-test-fold", body, "txt")
        .await
        .unwrap();
    let second = write_artifact("letcode-test-fold", body, "txt")
        .await
        .unwrap();
    assert_eq!(first, second);
    let written = tokio::fs::read(&first).await.unwrap();
    assert_eq!(written, body);
    let _ = tokio::fs::remove_file(&first).await;
    let _ = tokio::fs::remove_dir(std::env::temp_dir().join("letcode-test-fold")).await;
}
