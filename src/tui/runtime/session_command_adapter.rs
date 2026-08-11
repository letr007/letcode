//! TUI adapter that applies frontend-neutral [`SessionCommand`]s to the
//! session-owned engine ingress.
//!
//! This adapter keeps TUI-only view normalization out of the public engine
//! surface while preserving the staged session-engine migration.

use anyhow::{Result, bail};

use crate::session::{
    ActiveTurnCommandDisposition, SessionCommand, SessionCommandHandler, SessionEngineIngress,
};

use crate::tui::state::ToastKind;

use super::{TuiRuntime, child_navigation_anchor};

const SESSION_ENGINE_UNAVAILABLE_MESSAGE: &str = "Session engine is no longer available";

/// Applies session commands by enqueueing frontend-neutral engine intent.
pub(super) struct TuiSessionCommandAdapter<'a> {
    runtime: &'a mut TuiRuntime,
    ingress: &'a SessionEngineIngress,
    allow_submit_family: bool,
}

impl<'a> TuiSessionCommandAdapter<'a> {
    pub(super) fn new(
        runtime: &'a mut TuiRuntime,
        ingress: &'a SessionEngineIngress,
        allow_submit_family: bool,
    ) -> Self {
        Self {
            runtime,
            ingress,
            allow_submit_family,
        }
    }

    fn submit(&mut self, command: SessionCommand) -> Result<()> {
        let active_turn = self.runtime.has_active_or_pending_session_turn();
        let pending_mcp_server = match &command {
            SessionCommand::ToggleMcpServer(server_name) => Some(server_name.clone()),
            _ => None,
        };
        let deferred = active_turn
            && matches!(
                command.active_turn_disposition(),
                ActiveTurnCommandDisposition::Defer
            );
        let rejected = active_turn
            && matches!(
                command.active_turn_disposition(),
                ActiveTurnCommandDisposition::Reject
            );
        if rejected {
            self.runtime
                .show_toast("Turn still running", ToastKind::Info);
            return Ok(());
        }
        if self.ingress.submit(command.clone()).is_err() {
            if let Some(server_name) = pending_mcp_server {
                self.runtime.clear_mcp_server_updating(&server_name);
            }
            bail!(SESSION_ENGINE_UNAVAILABLE_MESSAGE);
        }
        if deferred {
            self.runtime.project_deferred_setting(&command);
            self.runtime
                .show_toast("Change queued for after the current turn", ToastKind::Info);
        }
        Ok(())
    }

    fn request_interrupt(&mut self) -> Result<()> {
        if self.ingress.request_interrupt().is_err() {
            bail!(SESSION_ENGINE_UNAVAILABLE_MESSAGE);
        }
        Ok(())
    }
}

impl SessionCommandHandler for TuiSessionCommandAdapter<'_> {
    fn handle(&mut self, command: SessionCommand) -> Result<()> {
        match command {
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact
                if !self.allow_submit_family =>
            {
                // Mouse path must not start turns or compaction.
                Ok(())
            }
            SessionCommand::ViewChild {
                navigation,
                anchor_child_session_id: _,
            } => {
                let anchor_child_session_id = child_navigation_anchor(self.runtime.state());
                self.submit(SessionCommand::ViewChild {
                    navigation,
                    anchor_child_session_id,
                })
            }
            SessionCommand::Interrupt => self.request_interrupt(),
            command => self.submit(command),
        }
    }
}
