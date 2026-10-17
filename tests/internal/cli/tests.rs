use super::*;
use crate::permission::PermissionApproval;

fn cli_view() -> CliView {
    CliView {
        session_id: "session".into(),
        model_id: "provider/model".into(),
        model_label: "Model".into(),
        permission_mode_label: "default".into(),
        reasoning_effort: None,
    }
}

fn project_cli_event(view: &mut CliView, event: SessionTransportEvent) -> EventMarker {
    present_transport_event(
        event,
        OutputMode::FinalOnly,
        false,
        &mut HashMap::new(),
        &mut false,
        view,
        &mut String::new(),
        &mut None,
        &mut Vec::new(),
    )
    .expect("event projection")
}

#[test]
fn permission_mode_changes_only_on_authoritative_event() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::PermissionModeChanged {
            mode: "safe".into(),
        },
    );

    assert!(marker.permission_mode_changed);
    assert_eq!(view.permission_mode_label, "safe");
}

#[test]
fn permission_error_does_not_change_cli_state() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::Error(crate::session::ErrorEvent::new("failed")),
    );

    assert!(marker.error);
    assert_eq!(view.permission_mode_label, "default");
}

#[test]
fn reasoning_effort_changes_only_on_authoritative_event() {
    let mut view = cli_view();
    let marker = project_cli_event(
        &mut view,
        SessionTransportEvent::ReasoningEffortChanged {
            effort: ModelReasoningEffort::High,
        },
    );

    assert!(marker.reasoning_effort_changed);
    assert_eq!(view.reasoning_effort, Some(ModelReasoningEffort::High));
}

#[test]
fn terminal_permission_input_parses_once_always_and_denial() {
    assert_eq!(
        permission_approval_from_input("o", true),
        PermissionApproval::AllowOnce
    );
    assert_eq!(
        permission_approval_from_input("a", true),
        PermissionApproval::AllowAlways
    );
    assert_eq!(
        permission_approval_from_input("always", false),
        PermissionApproval::Deny
    );
    assert_eq!(
        permission_approval_from_input("n", true),
        PermissionApproval::Deny
    );
}
