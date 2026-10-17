use super::validate_workflow_todos;
use serde_json::json;

#[test]
fn accepts_multiple_simultaneous_status_changes_in_one_snapshot() {
    let snapshot = json!({
        "items": [
            {"id": "investigate", "content": "Investigate issue", "status": "completed"},
            {"id": "implement", "content": "Implement fix", "status": "in_progress"},
            {"id": "validate", "content": "Validate fix", "status": "pending"}
        ]
    });

    assert!(validate_workflow_todos(&snapshot).is_ok());
}

#[test]
fn accepts_multiple_in_progress_items() {
    let snapshot = json!({
        "items": [
            {"id": "implement", "content": "Implement fix", "status": "in_progress"},
            {"id": "review", "content": "Review fix", "status": "in_progress"}
        ]
    });

    assert!(validate_workflow_todos(&snapshot).is_ok());
}
