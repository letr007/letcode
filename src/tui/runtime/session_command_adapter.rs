//! TUI adapter that applies frontend-neutral [`SessionCommand`]s to the
//! session-owned engine ingress.
//!
//! This adapter keeps TUI-only view normalization out of the public engine
//! surface while preserving the staged session-engine migration.

use anyhow::{Result, bail};

use crate::session::{SessionCommand, SessionCommandHandler, SessionEngineIngress};

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
        if self.ingress.submit(command).is_err() {
            bail!(SESSION_ENGINE_UNAVAILABLE_MESSAGE);
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
