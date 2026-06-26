use tokio::sync::mpsc;

use super::{
    ErrorEvent, InterruptRequest, RunnerCommand, RunnerEvent, RuntimeCommand, TuiRuntime,
    child_navigation_anchor,
};

const RUNNER_UNAVAILABLE_MESSAGE: &str = "TUI runner task is no longer available";

pub(super) fn dispatch_command(
    runtime: &mut TuiRuntime,
    command: RuntimeCommand,
    prompt_tx: &mpsc::UnboundedSender<RunnerCommand>,
    cancel_tx: &mpsc::UnboundedSender<InterruptRequest>,
    allow_submit_family: bool,
) {
    match command {
        RuntimeCommand::SubmitPrompt(prompt) if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::Prompt(prompt));
        }
        RuntimeCommand::DelegateSubagent { agent_name, task } if allow_submit_family => {
            send_prompt(
                runtime,
                prompt_tx,
                RunnerCommand::DelegateSubagent { agent_name, task },
            );
        }
        RuntimeCommand::Compact if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::Compact);
        }
        RuntimeCommand::ShowBranchTree if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::ShowBranchTree);
        }
        RuntimeCommand::ListBranches if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::ListBranches);
        }
        RuntimeCommand::CreateBranch { label } if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::CreateBranch { label });
        }
        RuntimeCommand::CheckoutBranch(branch_id) if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::CheckoutBranch(branch_id));
        }
        RuntimeCommand::SetPermissionMode(mode) if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::SetPermissionMode(mode));
        }
        RuntimeCommand::SetModel(model) if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::SetModel(model));
        }
        RuntimeCommand::SetReasoningEffort(effort) if allow_submit_family => {
            send_prompt(
                runtime,
                prompt_tx,
                RunnerCommand::SetReasoningEffort(effort),
            );
        }
        RuntimeCommand::ResumeSession(session_id) if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::ResumeSession(session_id));
        }
        RuntimeCommand::NewSession if allow_submit_family => {
            send_prompt(runtime, prompt_tx, RunnerCommand::NewSession);
        }
        RuntimeCommand::ViewChild(navigation) => {
            let anchor_child_session_id = child_navigation_anchor(runtime.state());
            send_prompt(
                runtime,
                prompt_tx,
                RunnerCommand::ViewChild {
                    navigation,
                    anchor_child_session_id,
                },
            );
        }
        RuntimeCommand::ViewParent => {
            send_prompt(runtime, prompt_tx, RunnerCommand::ViewParent);
        }
        RuntimeCommand::Interrupt => {
            if cancel_tx.send(runtime.build_interrupt_request()).is_err() {
                handle_runner_unavailable(runtime);
            }
        }
        RuntimeCommand::SubmitPrompt(_)
        | RuntimeCommand::DelegateSubagent { .. }
        | RuntimeCommand::Compact
        | RuntimeCommand::ShowBranchTree
        | RuntimeCommand::ListBranches
        | RuntimeCommand::CreateBranch { .. }
        | RuntimeCommand::CheckoutBranch(_)
        | RuntimeCommand::SetPermissionMode(_)
        | RuntimeCommand::SetModel(_)
        | RuntimeCommand::SetReasoningEffort(_)
        | RuntimeCommand::ResumeSession(_)
        | RuntimeCommand::NewSession => {}
    }
}

fn send_prompt(
    runtime: &mut TuiRuntime,
    prompt_tx: &mpsc::UnboundedSender<RunnerCommand>,
    command: RunnerCommand,
) {
    if prompt_tx.send(command).is_err() {
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
    use crate::tui::{AppPhase, TuiState};

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
        let (prompt_tx, prompt_rx) = mpsc::unbounded_channel();
        drop(prompt_rx);
        let (cancel_tx, _cancel_rx) = mpsc::unbounded_channel();

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SetModel("gpt-test".into()),
            &prompt_tx,
            &cancel_tx,
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
        let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel();
        let (cancel_tx, _cancel_rx) = mpsc::unbounded_channel();

        dispatch_command(
            &mut runtime,
            RuntimeCommand::SubmitPrompt("hello".into()),
            &prompt_tx,
            &cancel_tx,
            false,
        );

        assert!(prompt_rx.try_recv().is_err());
        assert!(runtime.state().timeline.items().is_empty());
    }
}
