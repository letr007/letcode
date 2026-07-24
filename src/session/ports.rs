//! Frontend-neutral ports for session backends.
//!
//! These traits define how a session implementation talks to the outside world
//! without depending on any UI crate. TUI/CLI/GUI provide adapters.

use crate::session::command::SessionCommand;
use crate::session::event::SessionEvent;

/// Sink for outbound session events (turn/stream/tool/lifecycle).
pub trait SessionEventSink {
    fn emit(&self, event: SessionEvent);
}

impl SessionEventSink for tokio::sync::mpsc::UnboundedSender<SessionEvent> {
    fn emit(&self, event: SessionEvent) {
        let _ = self.send(event);
    }
}

/// Inbound command application. Full turn execution may still be hosted by
/// [`crate::session::runner::AgentRunner`] until the engine absorbs it.
pub trait SessionCommandHandler {
    fn handle(&mut self, command: SessionCommand) -> anyhow::Result<()>;
}

/// Marker for the session backend boundary. Command execution and event emission
/// are supplied by adapters during the migration.
#[derive(Debug, Default)]
pub struct SessionPorts;
