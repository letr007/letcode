use super::validate_workflow_todos;
use serde_json::json;

#[test]
fn accepts_multiple_statuses_in_one_snapshot() {
    let snapshot = json!({
        "items": [
            {"id": "investigate", "content": "Investigate issue", "status": "completed"},
            {"id": "implement", "content": "Implement fix", "status": "in_progress"},
            {"id": "validate", "content": "Validate fix", "status": "pending"}
        ]
    });

    assert!(validate_workflow_todos(&snapshot).is_ok());
}
