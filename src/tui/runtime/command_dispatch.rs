use tokio::sync::mpsc;

use super::{
    ErrorEvent, RunnerCommand, RunnerControl, RunnerEvent, RuntimeCommand, TuiRuntime,
    child_navigation_anchor,
};

const RUNNER_UNAVAILABLE_MESSAGE: &str = "TUI runner task is no longer available";

pub(super) fn dispatch_command(
    runtime: &mut TuiRuntime,
    command: RuntimeCommand,
    control_tx: &mpsc::UnboundedSender<RunnerControl>,
    allow_submit_family: bool,
) {
    match command {
        RuntimeCommand::SubmitPrompt(prompt) if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::Prompt(prompt));
        }
        RuntimeCommand::DelegateSubagent { agent_name, task } if allow_submit_family => {
            send_command(
                runtime,
                control_tx,
                RunnerCommand::DelegateSubagent { agent_name, task },
            );
        }
        RuntimeCommand::Compact if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::Compact);
        }
        RuntimeCommand::ShowBranchTree if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::ShowBranchTree);
        }
        RuntimeCommand::ListBranches if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::ListBranches);
        }
        RuntimeCommand::SetPermissionMode(mode) if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::SetPermissionMode(mode));
        }
        RuntimeCommand::SetModel(model) if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::SetModel(model));
        }
        RuntimeCommand::SetReasoningEffort(effort) if allow_submit_family => {
            send_command(
                runtime,
                control_tx,
                RunnerCommand::SetReasoningEffort(effort),
            );
        }
        RuntimeCommand::ResumeSession(session_id) if allow_submit_family => {
            send_command(
                runtime,
                control_tx,
                RunnerCommand::ResumeSession(session_id),
            );
        }
        RuntimeCommand::NewSession if allow_submit_family => {
            send_command(runtime, control_tx, RunnerCommand::NewSession);
        }
        RuntimeCommand::ToggleMcpServer(server_name) => {
            send_command(
                runtime,
                control_tx,
                RunnerCommand::ToggleMcpServer(server_name),
            );
        }
        RuntimeCommand::ViewChild(navigation) => {
            let anchor_child_session_id = child_navigation_anchor(runtime.state());
            send_command(
                runtime,
                control_tx,
                RunnerCommand::ViewChild {
                    navigation,
                    anchor_child_session_id,
                },
            );
        }
        RuntimeCommand::ViewParent => {
            send_command(runtime, control_tx, RunnerCommand::ViewParent);
        }
        RuntimeCommand::Interrupt => {
            if control_tx
                .send(RunnerControl::Interrupt(runtime.build_interrupt_request()))
                .is_err()
            {
                handle_runner_unavailable(runtime);
            }
        }
        RuntimeCommand::SubmitPrompt(_)
        | RuntimeCommand::DelegateSubagent { .. }
        | RuntimeCommand::Compact
        | RuntimeCommand::ShowBranchTree
        | RuntimeCommand::ListBranches
        | RuntimeCommand::SetPermissionMode(_)
        | RuntimeCommand::SetModel(_)
        | RuntimeCommand::SetReasoningEffort(_)
        | RuntimeCommand::ResumeSession(_)
        | RuntimeCommand::NewSession => {}
    }
}

fn send_command(
    runtime: &mut TuiRuntime,
    control_tx: &mpsc::UnboundedSender<RunnerControl>,
    command: RunnerCommand,
) {
    if control_tx.send(RunnerControl::Command(command)).is_err() {
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
}
