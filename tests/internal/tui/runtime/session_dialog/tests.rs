use super::*;

fn summary(id: &str, last_timestamp_ms: u128, workspace: Option<&str>) -> SessionSummary {
    SessionSummary {
        session_id: id.into(),
        record_count: 1,
        first_timestamp_ms: Some(last_timestamp_ms),
        last_timestamp_ms: Some(last_timestamp_ms),
        model: Some("test/model".into()),
        title: Some(format!("title {id}")),
        last_user_summary: None,
        last_assistant_summary: None,
        workspace: workspace.map(str::to_string),
    }
}

fn ids(items: &[DialogItem]) -> Vec<&str> {
    items.iter().map(|item| item.id.as_str()).collect()
}

fn sections(items: &[DialogItem]) -> Vec<&str> {
    items
        .iter()
        .map(|item| item.section.as_deref().unwrap_or_default())
        .collect()
}

#[test]
fn workspace_scope_lists_only_the_running_workspace() {
    let sessions = vec![
        summary("current", 300, Some("/work/letcode")),
        summary("other", 200, Some("/work/other")),
        summary("unrecorded", 100, None),
    ];

    let items = session_dialog_items(
        &sessions,
        Some("/work/letcode"),
        SessionPickerScope::Workspace,
        "Unknown project",
    );

    assert_eq!(ids(&items), ["current"]);
}

#[test]
fn every_workspace_scope_leads_with_the_running_project() {
    let sessions = vec![
        summary("newest", 400, Some("/work/other")),
        summary("unrecorded", 300, None),
        summary("current", 200, Some("/work/letcode")),
        summary("older", 100, Some("/work/letcode")),
    ];

    let items = session_dialog_items(
        &sessions,
        Some("/work/letcode"),
        SessionPickerScope::All,
        "Unknown project",
    );

    assert_eq!(ids(&items), ["current", "older", "newest", "unrecorded"]);
    assert_eq!(
        sections(&items),
        ["letcode", "letcode", "other", "Unknown project"]
    );
}

#[test]
fn project_headings_widen_when_directory_names_collide() {
    let sessions = vec![
        summary("work", 200, Some("/work/api")),
        summary("oss", 100, Some("/oss/api")),
    ];

    let items = session_dialog_items(
        &sessions,
        Some("/elsewhere"),
        SessionPickerScope::All,
        "Unknown project",
    );

    assert_eq!(sections(&items), ["work/api", "oss/api"]);
}
