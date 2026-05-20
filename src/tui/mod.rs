#![allow(unused_imports, dead_code)]

//! Focused Ratatui/Crossterm TUI modules.
//!
//! The TUI is intentionally split into typed events, state/timeline models, pure
//! rendering, input mapping, terminal guards, runtime orchestration, and an agent
//! runner bridge. That keeps the view layer out of OpenAI/tool/transcript and
//! permission business logic, and avoids recreating the older `letcode-old`
//! implementation's monolithic TUI structure.
//!
//! Entry is intentionally explicit from `src/main.rs`: the default executable path
//! stays in the original line-based CLI, while `--tui` and `tui` opt into this
//! module's runtime. The old sibling project is only a visual and UX reference,
//! especially for the dark theme direction, not a blueprint for architecture.

pub mod events;
pub mod input;
pub mod presentation;
pub mod render;
pub mod runner;
pub mod runtime;
pub mod state;
pub mod terminal;
pub mod theme;
pub mod timeline;

pub use events::{
    AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
pub use input::{InputAction, apply_edit_action, map_key_event};
pub use presentation::{PresentationPolicy, ToolPresentation, ToolPresentationContext};
pub use render::render;
pub use runner::{
    AgentRunner, PermissionResponse, RunnerEvent, RunnerEventSender, RunnerPermissionRequest,
};
pub use runtime::{NoopDrawer, RuntimeCommand, RuntimeDrawer, TuiRuntime, run_tui};
pub use state::{AppPhase, FooterStatus, TuiState};
pub use terminal::{OwnedTerminal, TerminalGuard, TuiTerminal};
pub use theme::Theme;
pub use timeline::{
    DisplayBlock, ErrorView, MessageRole, MessageView, NoticeView, PermissionPromptStatus,
    PermissionView, Timeline, TimelineItem, ToolExecutionStatus, ToolView,
};
