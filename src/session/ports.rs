//! Frontend-neutral ports for session backends.
//!
//! These traits define how a session implementation talks to the outside world
//! without depending on any UI crate. TUI/CLI/GUI provide adapters.

use crate::session::command::SessionCommand;

/// Inbound command application. Full turn execution may still be hosted by
/// [`crate::session::runner::AgentRunner`] until the engine absorbs it.
///
/// Frontends implement this trait (for example the TUI control-channel adapter)
/// so presentation code can stay free of session-transport-private transport details.
/// Idle session work is increasingly owned by
/// [`crate::session::SessionCoordinator`].
pub trait SessionCommandHandler {
    fn handle(&mut self, command: SessionCommand) -> anyhow::Result<()>;
}
