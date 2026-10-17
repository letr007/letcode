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
#[path = "../../../tests/internal/tui/runtime/permission_lifecycle/tests.rs"]
mod tests;
