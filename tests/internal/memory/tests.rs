use super::*;
use serde_json::json;

#[test]
fn validates_memory_recall_query() {
    let query = validate_memory_recall_query(&json!({
        "query": " parser ",
        "paths": ["src/parser.rs"],
        "kinds": ["decision", "validation"],
        "statuses": ["active", "useful"],
        "limit": 3
    }))
    .unwrap();
    assert_eq!(query.query.as_deref(), Some("parser"));
    assert_eq!(query.paths, vec!["src/parser.rs"]);
    assert_eq!(query.kinds.len(), 2);
    assert_eq!(query.statuses.len(), 2);
    assert_eq!(query.limit, 3);
    assert!(validate_memory_recall_query(&json!({"limit": 99})).is_err());
}

#[test]
fn normalizes_empty_query_to_none_without_relaxing_validation() {
    let null_query = validate_memory_recall_query(&json!({"query": null})).unwrap();
    let empty_query = validate_memory_recall_query(&json!({"query": ""})).unwrap();
    let whitespace_query = validate_memory_recall_query(&json!({"query": "  \t\n"})).unwrap();

    assert_eq!(empty_query, null_query);
    assert_eq!(whitespace_query, null_query);
    assert!(validate_memory_recall_query(&json!({"query": 1})).is_err());
    assert!(validate_memory_recall_query(&json!({"paths": [""]})).is_err());
    assert!(validate_memory_recall_query(&json!({"paths": [" "]})).is_err());
}
