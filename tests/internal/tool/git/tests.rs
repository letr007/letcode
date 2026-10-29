use super::*;
use serde_json::json;

#[test]
fn ref_accepts_common_revisions() {
    for reference in ["HEAD", "origin/main", "HEAD~2", "v0.26.3", "a..b"] {
        assert_eq!(safe_ref_arg(reference).unwrap(), reference);
    }
}

#[test]
fn ref_rejects_option_like_and_spaced_values() {
    for reference in ["--output=/tmp/x", "-c", "HEAD main", "HEAD\tmain"] {
        assert!(safe_ref_arg(reference).is_err(), "{reference}");
    }
}

#[test]
fn path_rejects_traversal_and_absolute_paths() {
    let mut args = Vec::new();
    assert!(push_path(&mut args, Some("../secret")).is_err());

    let mut args = Vec::new();
    assert!(push_path(&mut args, Some("/etc/passwd")).is_err());

    let mut args = Vec::new();
    push_path(&mut args, Some("src/main.rs")).unwrap();
    let normalized = std::path::Path::new("src")
        .join("main.rs")
        .to_string_lossy()
        .into_owned();
    assert_eq!(args, ["--".to_string(), normalized]);
}

#[tokio::test]
async fn blame_requires_a_path() {
    let error = GitTool
        .execute(json!({ "action": "blame" }))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("blame requires a path"));
}

#[tokio::test]
async fn unknown_action_is_rejected() {
    let error = GitTool
        .execute(json!({ "action": "commit" }))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unsupported git action"));
}
