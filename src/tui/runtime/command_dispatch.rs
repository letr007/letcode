use super::{
    ErrorEvent, RuntimeCommand, SESSION_ENGINE_UNAVAILABLE_MESSAGE, SessionTransportEvent,
    TuiRuntime, session_command_adapter::TuiSessionCommandAdapter,
};
use crate::session::{SessionCommandHandler, SessionEngineIngress};

pub(super) fn dispatch_command(
    runtime: &mut TuiRuntime,
    command: RuntimeCommand,
    ingress: &SessionEngineIngress,
    allow_submit_family: bool,
) {
    let mut adapter = TuiSessionCommandAdapter::new(runtime, ingress, allow_submit_family);
    if adapter.handle(command).is_err() {
        handle_session_engine_unavailable(runtime);
    }
}

fn handle_session_engine_unavailable(runtime: &mut TuiRuntime) {
    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        SESSION_ENGINE_UNAVAILABLE_MESSAGE,
    )));
    runtime.apply_session_transport_event(SessionTransportEvent::TurnFinished {
        outcome: crate::session::TurnOutcome::Aborted,
    });
    runtime.clear_unaccepted_queued_prompt();
}

#[cfg(test)]
#[path = "../../../tests/internal/tui/runtime/command_dispatch/integration_tests.rs"]
mod integration_tests;
