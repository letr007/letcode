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

pub mod catalog;
pub mod components;
pub mod events;
pub mod input;
pub mod markdown;
pub(crate) mod math;
pub mod measure;
pub(crate) mod mermaid;
pub mod preferences;
pub mod presentation;
pub mod render;
pub mod runtime;
pub mod selection;
pub mod slash;
pub mod state;
pub mod surface;
pub mod terminal;
pub mod theme;
pub mod theme_file;
pub mod timeline;
pub mod transcript_ratatui;
pub(crate) mod transcript_read_model;
pub mod transcript_render;

#[allow(unused_imports)]
pub use events::{
    AssistantDeltaEvent, ErrorEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ReasoningDeltaEvent, ReasoningDoneEvent, SessionEvent,
    ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
#[allow(unused_imports)]
pub use input::{InputAction, apply_edit_action, map_key_event};
#[allow(unused_imports)]
pub use presentation::{PresentationPolicy, ToolPresentation};
#[allow(unused_imports)]
pub use render::render;
#[allow(unused_imports)]
pub use runtime::{RuntimeCommand, RuntimeDrawer, StartupToast, TuiRuntime, run_tui};
#[allow(unused_imports)]
pub use state::{AppPhase, RetryNoticeState, SelectionAnchor, TextSelection, TuiState};
#[allow(unused_imports)]
pub use terminal::{OwnedTerminal, TerminalGuard, TuiTerminal};
#[allow(unused_imports)]
pub use theme::{Theme, ThemeName};
#[allow(unused_imports)]
pub use timeline::{
    AutoReviewDecisionView, ErrorView, MessageRole, MessageView, PermissionPromptStatus,
    PermissionView, ReasoningView, Timeline, TimelineItem, ToolExecutionStatus, ToolView,
};
