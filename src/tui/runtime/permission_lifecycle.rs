use crate::session::RunnerPermissionRequest;
use crate::tui::events::{PermissionRequestEvent, PermissionResolutionEvent, SessionEvent};
use crate::tui::timeline::PermissionView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermissionOrigin {
    Parent,
    Child { child_session_id: String },
}

#[derive(Debug, Clone)]
pub(crate) struct PendingPermission {
    view: PermissionView,
    handle: Option<RunnerPermissionRequest>,
    origin: PermissionOrigin,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PermissionLifecycleController {
    pending: Option<PendingPermission>,
}

impl PermissionLifecycleController {
    #[cfg(test)]
    pub(crate) fn handle(&self) -> Option<&RunnerPermissionRequest> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.handle.as_ref())
    }

    pub(crate) fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn projection(&self) -> Option<PermissionView> {
        self.pending.as_ref().map(|pending| pending.view.clone())
    }

    pub(crate) fn begin_parent(
        &mut self,
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    ) -> Result<(), RunnerPermissionRequest> {
        self.begin(event, handle, PermissionOrigin::Parent)
    }

    pub(crate) fn begin_child(
        &mut self,
        child_session_id: String,
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    ) -> Result<(), RunnerPermissionRequest> {
        self.begin(event, handle, PermissionOrigin::Child { child_session_id })
    }

    fn begin(
        &mut self,
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
        origin: PermissionOrigin,
    ) -> Result<(), RunnerPermissionRequest> {
        if self.pending.is_some() {
            return Err(handle);
        }

        self.pending = Some(PendingPermission {
            view: PermissionView::from_request(event),
            handle: Some(handle),
            origin,
        });
        Ok(())
    }

    pub(crate) fn take_handle(&mut self) -> Option<RunnerPermissionRequest> {
        self.pending
            .as_mut()
            .and_then(|pending| pending.handle.take())
    }

    pub(crate) fn clear(&mut self) {
        self.pending = None;
    }

    pub(crate) fn clear_if_parent(&mut self) {
        if self.belongs_to_parent() {
            self.clear();
        }
    }

    pub(crate) fn belongs_to_parent(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.is_parent())
    }

    #[cfg(test)]
    pub(crate) fn child_session_id(&self) -> Option<&str> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.child_session_id())
    }

    pub(crate) fn matches_call(&self, call_id: &str, child_session_id: Option<&str>) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.matches_call(call_id, child_session_id))
    }

    pub(crate) fn clears_for_child_event(
        &self,
        child_session_id: &str,
        event: &SessionEvent,
    ) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.clears_for_child_event(child_session_id, event))
    }
}

impl PendingPermission {
    pub(crate) fn call_id(&self) -> &str {
        &self.view.call_id
    }

    pub(crate) fn is_parent(&self) -> bool {
        matches!(self.origin, PermissionOrigin::Parent)
    }

    pub(crate) fn child_session_id(&self) -> Option<&str> {
        match &self.origin {
            PermissionOrigin::Parent => None,
            PermissionOrigin::Child { child_session_id } => Some(child_session_id.as_str()),
        }
    }

    fn matches_call(&self, call_id: &str, child_session_id: Option<&str>) -> bool {
        self.call_id() == call_id && self.child_session_id() == child_session_id
    }

    fn clears_for_child_event(&self, child_session_id: &str, event: &SessionEvent) -> bool {
        match event {
            SessionEvent::PermissionResolved(PermissionResolutionEvent { call_id, .. }) => {
                self.matches_call(call_id, Some(child_session_id))
            }
            SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted => {
                self.child_session_id() == Some(child_session_id)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
