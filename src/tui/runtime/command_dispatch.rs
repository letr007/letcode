use super::{
    ErrorEvent, RuntimeCommand, SessionTransportEvent, TuiRuntime,
    session_command_adapter::TuiSessionCommandAdapter,
};
use crate::session::{SessionCommandHandler, SessionEngineIngress};

const SESSION_ENGINE_UNAVAILABLE_MESSAGE: &str = "Session engine is no longer available";

pub(super) fn dispatch_command(
    runtime: &mut TuiRuntime,
    command: RuntimeCommand,
    ingress: &SessionEngineIngress,
    allow_submit_family: bool,
) {
    let mut adapter = TuiSessionCommandAdapter::new(runtime, ingress, allow_submit_family);
    if adapter.handle(command).is_err() {
        // Channel closed: surface the same unavailable path as before.
        handle_session_engine_unavailable(runtime);
    }
}

fn handle_session_engine_unavailable(runtime: &mut TuiRuntime) {
    runtime.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
        SESSION_ENGINE_UNAVAILABLE_MESSAGE,
    )));
    runtime.apply_session_transport_event(SessionTransportEvent::Done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SessionEngine, engine::SessionEngineControl};
    use crate::tui::{AppPhase, TuiState, map_key_event};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        )
    }

    #[test]
    fn dispatch_reports_session_engine_unavailable_for_key_path() {
        let mut runtime = runtime();
        let (engine, ingress, _egress) = SessionEngine::new();
        drop(engine);

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SetModel("gpt-test".into()),
            &ingress,
            true,
        );

        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, crate::tui::TimelineItem::Error(error) if error.message == SESSION_ENGINE_UNAVAILABLE_MESSAGE)));
    }

    #[tokio::test]
    async fn dispatch_ignores_submit_family_for_mouse_path() {
        let mut runtime = runtime();
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SubmitPrompt("hello".into()),
            &ingress,
            false,
        );

        assert!(engine.try_recv_control().is_err());
        assert!(runtime.state().timeline.items().is_empty());
    }

    #[tokio::test]
    async fn double_escape_dispatches_frontend_neutral_interrupt() {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state.phase = AppPhase::Error;
        let (mut engine, ingress, _egress) = SessionEngine::new();
        let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

        let first = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("first Esc is accepted");
        assert_eq!(first, None);
        let second = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("second Esc is accepted")
            .expect("second Esc requests interruption");

        dispatch_command(&mut runtime, second, &ingress, true);
        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Interrupt)
        ));
    }

    #[tokio::test]
    async fn dispatch_maps_session_command_through_engine_ingress() {
        let mut runtime = runtime();
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(&mut runtime, RuntimeCommand::Compact, &ingress, true);

        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Command(_))
        ));
    }
}
