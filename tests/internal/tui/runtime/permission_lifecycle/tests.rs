use super::*;
use crate::tui::events::{ErrorEvent, PermissionDecision};
use tokio::sync::oneshot;

fn handle() -> RunnerPermissionRequest {
    let (tx, _rx) = oneshot::channel();
    RunnerPermissionRequest::new(tx)
}

#[test]
fn matches_call_respects_parent_and_child_origin() {
    let mut controller = PermissionLifecycleController::default();
    controller
        .begin_parent(
            PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
            handle(),
        )
        .expect("begin parent permission");

    assert!(controller.matches_call("call-1", None));
    assert!(!controller.matches_call("call-1", Some("child-1")));

    controller.clear();
    controller
        .begin_child(
            "child-1".into(),
            PermissionRequestEvent::new("call-2", "shell__exec", "run cargo test"),
            handle(),
        )
        .expect("begin child permission");

    assert!(controller.matches_call("call-2", Some("child-1")));
    assert!(!controller.matches_call("call-2", Some("other-child")));
    assert!(!controller.matches_call("call-2", None));
}

#[test]
fn child_event_clearing_only_clears_matching_child_permission() {
    let mut controller = PermissionLifecycleController::default();
    controller
        .begin_child(
            "child-1".into(),
            PermissionRequestEvent::new("call-2", "shell__exec", "run cargo test"),
            handle(),
        )
        .expect("begin child permission");

    assert!(controller.clears_for_child_event(
        "child-1",
        &SessionEvent::PermissionResolved(PermissionResolutionEvent {
            call_id: "call-2".into(),
            decision: PermissionDecision::Approved,
            reason: None,
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        })
    ));
    assert!(!controller.clears_for_child_event(
        "child-1",
        &SessionEvent::PermissionResolved(PermissionResolutionEvent {
            call_id: "other-call".into(),
            decision: PermissionDecision::Approved,
            reason: None,
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        })
    ));
    assert!(controller.clears_for_child_event(
        "child-1",
        &SessionEvent::Error(ErrorEvent::new("child failed"))
    ));
    assert!(!controller.clears_for_child_event("other-child", &SessionEvent::Interrupted));
}

#[test]
fn representative_child_permission_lifecycle_flow_matches_and_clears() {
    let mut controller = PermissionLifecycleController::default();
    controller
        .begin_child(
            "child-7".into(),
            PermissionRequestEvent::new("call-7", "shell__exec", "cargo test"),
            handle(),
        )
        .expect("begin child permission");

    assert!(controller.is_pending());
    assert!(controller.matches_call("call-7", Some("child-7")));
    assert!(controller.clears_for_child_event(
        "child-7",
        &SessionEvent::PermissionResolved(PermissionResolutionEvent::approved("call-7"))
    ));

    controller.clear();

    assert!(!controller.is_pending());
    assert_eq!(controller.projection(), None);
}
