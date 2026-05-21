//! Focused Ratatui/Crossterm TUI modules.
//!
//! The TUI is intentionally split into typed events, state/timeline models, pure
//! rendering, input mapping, terminal guards, runtime orchestration, and an agent
//! runner bridge. That keeps the view layer out of OpenAI/tool/transcript and
//! permission business logic, and avoids recreating the older `letcode-old`
//! implementation's monolithic TUI structure.
//!
//! Entry is coordinated from `src/main.rs`: the default executable path starts
//! this TUI, while `--cli` and `cli` keep the original line-based interface
//! available explicitly. The old sibling project is only a visual and UX
//! reference, especially for the dark theme direction, not a blueprint for
//! architecture.

pub mod components;
pub mod events;
pub mod input;
pub mod measure;
pub mod presentation;
pub mod render;
pub mod runner;
pub mod runtime;
pub mod state;
pub mod surface;
pub mod terminal;
pub mod theme;
pub mod timeline;

#[allow(unused_imports)]
pub use events::{
    AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
#[allow(unused_imports)]
pub use input::{InputAction, apply_edit_action, map_key_event};
#[allow(unused_imports)]
pub use presentation::{PresentationPolicy, ToolPresentation, ToolPresentationContext};
#[allow(unused_imports)]
pub use render::render;
#[allow(unused_imports)]
pub use runner::{
    AgentRunner, PermissionResponse, RunnerEvent, RunnerEventSender, RunnerPermissionRequest,
};
#[allow(unused_imports)]
pub use runtime::{NoopDrawer, RuntimeCommand, RuntimeDrawer, TuiRuntime, run_tui};
#[allow(unused_imports)]
pub use state::{AppPhase, FooterStatus, TuiState};
#[allow(unused_imports)]
pub use terminal::{OwnedTerminal, TerminalGuard, TuiTerminal};
#[allow(unused_imports)]
pub use theme::Theme;
#[allow(unused_imports)]
pub use timeline::{
    DisplayBlock, ErrorView, MessageRole, MessageView, NoticeView, PermissionPromptStatus,
    PermissionView, Timeline, TimelineItem, ToolExecutionStatus, ToolView,
};
