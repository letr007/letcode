//! TUI adapter that applies frontend-neutral [`SessionCommand`]s to the runner
//! control channel.
//!
//! This is the Phase C implementor of [`SessionCommandHandler`]: presentation
//! code builds `SessionCommand` / `RuntimeCommand`, and this adapter owns the
//! mapping onto the private `RunnerCommand` transport (including child-nav
//! anchors and interrupt requests).

use anyhow::{Result, bail};
use tokio::sync::mpsc;

use crate::session::{SessionCommand, SessionCommandHandler};

use super::{RunnerCommand, RunnerControl, TuiRuntime, child_navigation_anchor};

const RUNNER_UNAVAILABLE_MESSAGE: &str = "TUI runner task is no longer available";

/// Applies session commands by enqueueing runner control messages.
pub(super) struct TuiSessionCommandAdapter<'a> {
    runtime: &'a mut TuiRuntime,
    control_tx: &'a mpsc::UnboundedSender<RunnerControl>,
    allow_submit_family: bool,
}

impl<'a> TuiSessionCommandAdapter<'a> {
    pub(super) fn new(
        runtime: &'a mut TuiRuntime,
        control_tx: &'a mpsc::UnboundedSender<RunnerControl>,
        allow_submit_family: bool,
    ) -> Self {
        Self {
            runtime,
            control_tx,
            allow_submit_family,
        }
    }

    fn send_runner_command(&mut self, command: RunnerCommand) -> Result<()> {
        if self
            .control_tx
            .send(RunnerControl::Command(command))
            .is_err()
        {
            bail!(RUNNER_UNAVAILABLE_MESSAGE);
        }
        Ok(())
    }

    fn send_interrupt(&mut self) -> Result<()> {
        let request = self.runtime.build_interrupt_request();
        if self
            .control_tx
            .send(RunnerControl::Interrupt(request))
            .is_err()
        {
            bail!(RUNNER_UNAVAILABLE_MESSAGE);
        }
        Ok(())
    }
}

impl SessionCommandHandler for TuiSessionCommandAdapter<'_> {
    fn handle(&mut self, command: SessionCommand) -> Result<()> {
        match command {
            SessionCommand::SubmitPrompt(prompt) if self.allow_submit_family => {
                self.send_runner_command(RunnerCommand::Prompt(prompt))
            }
            SessionCommand::DelegateSubagent { agent_name, task } if self.allow_submit_family => {
                self.send_runner_command(RunnerCommand::DelegateSubagent { agent_name, task })
            }
            SessionCommand::Compact if self.allow_submit_family => {
                self.send_runner_command(RunnerCommand::Compact)
            }
            SessionCommand::ShowBranchTree => {
                self.send_runner_command(RunnerCommand::ShowBranchTree)
            }
            SessionCommand::ListBranches => self.send_runner_command(RunnerCommand::ListBranches),
            SessionCommand::SetPermissionMode(mode) => {
                self.send_runner_command(RunnerCommand::SetPermissionMode(mode))
            }
            SessionCommand::SetModel(model) => {
                self.send_runner_command(RunnerCommand::SetModel(model))
            }
            SessionCommand::SetReasoningEffort(effort) => {
                self.send_runner_command(RunnerCommand::SetReasoningEffort(effort))
            }
            SessionCommand::ResumeSession(session_id) => {
                self.send_runner_command(RunnerCommand::ResumeSession(session_id))
            }
            SessionCommand::NewSession => self.send_runner_command(RunnerCommand::NewSession),
            SessionCommand::ToggleMcpServer(server_name) => {
                self.send_runner_command(RunnerCommand::ToggleMcpServer(server_name))
            }
            SessionCommand::ViewChild {
                navigation,
                anchor_child_session_id: _,
            } => {
                let anchor_child_session_id = child_navigation_anchor(self.runtime.state());
                self.send_runner_command(RunnerCommand::ViewChild {
                    navigation,
                    anchor_child_session_id,
                })
            }
            SessionCommand::ViewParent => self.send_runner_command(RunnerCommand::ViewParent),
            SessionCommand::Interrupt => self.send_interrupt(),
            // Mouse path must not start turns or compaction.
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact => Ok(()),
        }
    }
}
