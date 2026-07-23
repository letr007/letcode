//! Session backend boundary for all frontends (TUI, line CLI, future GUI).
//!
//! ## Split intent
//!
//! - **Backend (`session`)**: turn execution, tools, permissions, subagents,
//!   transcript/session lifecycle. Frontends submit commands and observe
//!   results/events; they do not own agent internals.
//! - **Frontend (`tui`, CLI, GUI)**: presentation, input, layout, and local
//!   view-model state. Frontends may full-redraw; they should not re-implement
//!   session policy.
//!
//! ## Current surface
//!
//! Phase 1 of the split exposes the inbound command contract
//! ([`SessionCommand`]). Outbound events still flow through the existing
//! agent/TUI runner bridges; they will migrate here next without changing
//! product behavior.
//!
//! ```text
//!   TUI / CLI / GUI
//!        │ SessionCommand
//!        ▼
//!   session (engine boundary)
//!        │ agent / tools / transcript / subagent pool
//!        ▼
//!   model providers + local workspace
//! ```

pub mod command;

pub use command::SessionCommand;
