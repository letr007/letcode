use tokio::sync::mpsc;

use super::{
    ErrorEvent, RunnerControl, RunnerEvent, RuntimeCommand, TuiRuntime,
    session_command_adapter::TuiSessionCommandAdapter,
};
use crate::session::SessionCommandHandler;

const RUNNER_UNAVAILABLE_MESSAGE: &str = "TUI runner task is no longer available";

pub(super) fn dispatch_command(
    runtime: &mut TuiRuntime,
    command: RuntimeCommand,
    control_tx: &mpsc::UnboundedSender<RunnerControl>,
    allow_submit_family: bool,
) {
    let mut adapter = TuiSessionCommandAdapter::new(runtime, control_tx, allow_submit_family);
    if adapter.handle(command).is_err() {
        // Channel closed: surface the same unavailable path as before.
        handle_runner_unavailable(runtime);
    }
}

fn handle_runner_unavailable(runtime: &mut TuiRuntime) {
    runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new(
        RUNNER_UNAVAILABLE_MESSAGE,
    )));
    runtime.apply_runner_event(RunnerEvent::Done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{AppPhase, TuiState, map_key_event};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::super::{RunnerCommand, RunnerControl};

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = mpsc::unbounded_channel();
        TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        )
    }

    #[test]
    fn dispatch_reports_runner_unavailable_for_key_path() {
        let mut runtime = runtime();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        drop(control_rx);

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SetModel("gpt-test".into()),
            &control_tx,
            true,
        );

        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, crate::tui::TimelineItem::Error(error) if error.message == RUNNER_UNAVAILABLE_MESSAGE)));
    }

    #[test]
    fn dispatch_ignores_submit_family_for_mouse_path() {
        let mut runtime = runtime();
        let (control_tx, mut control_rx) = mpsc::unbounded_channel();

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SubmitPrompt("hello".into()),
            &control_tx,
            false,
        );

        assert!(control_rx.try_recv().is_err());
        assert!(runtime.state().timeline.items().is_empty());
    }

    #[test]
    fn double_escape_dispatches_cancel_while_error_projection_has_live_runner() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state.phase = AppPhase::Error;
        let (control_tx, mut control_rx) = mpsc::unbounded_channel();
        let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

        let first = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("first Esc is accepted");
        assert_eq!(first, None);
        let second = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("second Esc is accepted")
            .expect("second Esc requests interruption");

        dispatch_command(&mut runtime, second, &control_tx, true);
        assert!(matches!(
            control_rx.try_recv(),
            Ok(RunnerControl::Interrupt(_))
        ));
    }

    #[test]
    fn dispatch_maps_session_command_through_handler() {
        let mut runtime = runtime();
        let (control_tx, mut control_rx) = mpsc::unbounded_channel();

        dispatch_command(&mut runtime, RuntimeCommand::Compact, &control_tx, true);

        assert!(matches!(
            control_rx.try_recv(),
            Ok(RunnerControl::Command(RunnerCommand::Compact))
        ));
    }
}
