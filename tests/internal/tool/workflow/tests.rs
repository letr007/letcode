use super::validate_workflow_todos;
use serde_json::json;

#[test]
fn accepts_multiple_statuses_in_one_snapshot() {
    for items in [
        json!([
            {"id": "investigate", "content": "Investigate issue", "status": "completed"},
            {"id": "implement", "content": "Implement fix", "status": "in_progress"},
            {"id": "validate", "content": "Validate fix", "status": "pending"}
        ]),
        json!([
            {"id": "implement", "content": "Implement fix", "status": "in_progress"},
            {"id": "review", "content": "Review fix", "status": "in_progress"}
        ]),
    ] {
        assert!(validate_workflow_todos(&json!({ "items": items })).is_ok());
    }
}
