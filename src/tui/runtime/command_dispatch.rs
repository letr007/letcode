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
            Vec::new(),
            std::env::temp_dir(),
            std::env::temp_dir(),
        )
    }

    #[test]
    fn idle_compact_reaches_engine_and_returns_to_completed() {
        for phase in [AppPhase::Idle, AppPhase::Completed] {
            let mut runtime = runtime();
            runtime.state.phase = phase;
            runtime.state_mut().set_input("/compact");
            let command = runtime
                .handle_input_action(crate::tui::InputAction::Submit)
                .unwrap()
                .expect("idle compact produces a command");
            let (mut engine, ingress, _egress) = SessionEngine::new();

            dispatch_command(&mut runtime, command, &ingress, true);

            assert!(matches!(
                engine.try_recv_control(),
                Ok(SessionEngineControl::Command(
                    crate::session::engine::SessionEngineCommand::Compact
                ))
            ));
            assert!(runtime.session_turn_active);
            assert_eq!(runtime.state.phase, AppPhase::Running);
            assert_eq!(
                runtime.state.toast().map(|toast| toast.message.clone()),
                Some(runtime.state.t("runtime.compacting_context")),
            );

            runtime.apply_session_transport_event(SessionTransportEvent::CompactionStarted);
            runtime.apply_session_transport_event(SessionTransportEvent::CompactionNoProgress {
                blockers: Vec::new(),
            });
            runtime.apply_session_transport_event(SessionTransportEvent::Done);
            assert!(!runtime.has_active_or_pending_session_turn());
            assert_eq!(runtime.state.phase, AppPhase::Completed);
        }
    }

    #[test]
    fn idle_delegation_reaches_engine_before_projecting_running_state() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_input("@explorer inspect the workspace");
        let command = runtime
            .handle_input_action(crate::tui::InputAction::Submit)
            .unwrap()
            .expect("idle delegation produces a command");
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(&mut runtime, command, &ingress, true);

        assert!(matches!(
            engine.try_recv_control(),
            Ok(SessionEngineControl::Command(
                crate::session::engine::SessionEngineCommand::DelegateSubagent { agent_name, task }
            )) if agent_name == "explorer" && task == "inspect the workspace"
        ));
        assert!(runtime.session_turn_active);
        assert_eq!(runtime.state.phase, AppPhase::Running);
        assert_eq!(
            runtime
                .state
                .timeline
                .items()
                .iter()
                .filter(|item| { matches!(item, crate::tui::TimelineItem::Delegation(_)) })
                .count(),
            1
        );
    }

    #[test]
    fn compact_and_delegation_reject_active_turn_without_ending_it() {
        for input in ["/compact", "@explorer inspect the workspace"] {
            let mut runtime = runtime();
            runtime.session_turn_active = true;
            runtime.state.phase = AppPhase::Running;
            runtime.state_mut().set_input(input);
            assert!(
                runtime
                    .handle_input_action(crate::tui::InputAction::Submit)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(runtime.state.input_buffer, input);
            let (mut engine, ingress, _egress) = SessionEngine::new();
            let command = if input == "/compact" {
                crate::session::SessionCommand::Compact
            } else {
                crate::session::SessionCommand::DelegateSubagent {
                    agent_name: "explorer".into(),
                    task: "inspect the workspace".into(),
                }
            };
            dispatch_command(&mut runtime, command, &ingress, true);
            assert!(matches!(
                engine.try_recv_control(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ));
            assert!(runtime.session_turn_active);
            assert_eq!(runtime.state.phase, AppPhase::Running);
            assert_eq!(
                runtime.state.toast().map(|toast| toast.message.clone()),
                Some(runtime.state.t("runtime.turn_running"))
            );
        }
    }

    #[test]
    fn failed_compact_dispatch_does_not_leave_running_state() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/compact");
        let command = runtime
            .handle_input_action(crate::tui::InputAction::Submit)
            .unwrap()
            .unwrap();
        let (engine, ingress, _egress) = SessionEngine::new();
        drop(engine);

        dispatch_command(&mut runtime, command, &ingress, true);

        assert!(!runtime.has_active_or_pending_session_turn());
        assert_eq!(runtime.state.phase, AppPhase::Completed);
    }

    #[test]
    fn failed_resume_dispatch_clears_pending_state() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/resume session-1");
        let command = runtime
            .handle_input_action(crate::tui::InputAction::Submit)
            .expect("resume command succeeds")
            .expect("resume command is dispatched");
        assert!(runtime.session_resume_pending);

        let (engine, ingress, _egress) = SessionEngine::new();
        drop(engine);
        dispatch_command(&mut runtime, command, &ingress, true);

        assert!(!runtime.session_resume_pending);
        assert_eq!(runtime.state().phase, AppPhase::Completed);
    }

    #[tokio::test]
    async fn running_destructive_command_is_rejected_before_engine_ingress() {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state.phase = AppPhase::Running;
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(
            &mut runtime,
            crate::session::SessionCommand::NewSession,
            &ingress,
            true,
        );

        assert!(matches!(
            engine.try_recv_control(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let expected = runtime.state().t("runtime.turn_running");
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some(expected.as_str())
        );
    }

    #[tokio::test]
    async fn running_setting_dispatch_projects_pending_value_without_local_notice() {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state.phase = AppPhase::Running;
        let (mut engine, ingress, _egress) = SessionEngine::new();

        dispatch_command(
            &mut runtime,
            crate::session::SessionCommand::SetModel("provider/model".into()),
            &ingress,
            true,
        );

        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Command(
                crate::session::engine::SessionEngineCommand::SetModel(model)
            )) if model == "provider/model"
        ));
        assert!(runtime.state().toast().is_none());
        assert_eq!(
            runtime.state().pending_composer_settings.model,
            Some(("provider/model".into(), "provider/model".into()))
        );
        assert_eq!(runtime.state().model_id, "pending-runtime-model");
    }

    #[test]
    fn failed_mcp_dispatch_clears_updating_state() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_server_updating("docs".into(), true);
        let (engine, ingress, _egress) = SessionEngine::new();
        drop(engine);

        dispatch_command(
            &mut runtime,
            crate::session::SessionCommand::ToggleMcpServer("docs".into()),
            &ingress,
            true,
        );

        assert!(!runtime.state().mcp_updating.contains("docs"));
        assert_eq!(runtime.state().phase, AppPhase::Completed);
    }

    #[test]
    fn failed_setting_dispatch_does_not_project_pending_value() {
        let mut runtime = runtime();
        runtime.session_turn_active = true;
        runtime.state.phase = AppPhase::Running;
        let (engine, ingress, _egress) = SessionEngine::new();
        drop(engine);

        dispatch_command(
            &mut runtime,
            crate::session::SessionCommand::SetPermissionMode(
                crate::permission::PermissionMode::Safe,
            ),
            &ingress,
            true,
        );

        assert_eq!(
            runtime.state().pending_composer_settings.permission_mode,
            None
        );
        assert_eq!(runtime.state().phase, AppPhase::Completed);
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
}
