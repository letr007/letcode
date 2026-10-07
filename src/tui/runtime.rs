use std::collections::VecDeque;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use crossterm::event::{Event, KeyEventKind};
use tokio::sync::mpsc;

use crate::command::{
    ChildNavigation as SharedChildNavigation, CommandIntent, FakeCommand, PanelMode, ThemeCommand,
    ThoughtsDisplayMode, ToolsDisplayMode, TranscriptScrollbarMode, help_summary, parse_command,
};
use crate::mcp;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::session::archive::{self, ArchiveBudget, ArchiveRunReport, merged_session_summaries};
use crate::skills::SkillCard;
use crate::transcript::{SessionSummary, read_records, transcript_projection};
use crate::user_content::{UserImageAttachment, UserMessageSubmission};

use super::catalog::{mcp_dialog_items, mcp_tool_dialog_items, skill_dialog_items};
use super::events::{ErrorEvent, SessionEvent};
use super::input::{
    InputAction, apply_edit_action, map_key_event, map_mouse_event, map_paste_event,
};
use super::preferences::TuiPreferences;
use super::render;
use super::slash::{SlashCommandEntry, matching_completion_commands};
use super::state::{
    ConfigFieldRef, ContextDetailTarget, DialogItem, DialogKind, DialogState, PendingQuestionState,
    PermissionChoice, QuestionAdvance, SessionPickerScope, ToastKind, TranscriptClickTarget,
    TuiState,
};
use super::terminal::OwnedTerminal;
use super::theme::{Theme, ThemeName};
use super::theme_file::{
    CustomThemeInfo, bundled_theme_description, discover_custom_themes, ensure_bundled_themes,
    load_custom_theme, normalize_theme_id,
};
#[cfg(test)]
use crate::session::RunnerPermissionRequest;
use crate::session::runner::ModelCatalogUpdatedEvent;
use crate::session::{
    RunnerQuestionRequest, SessionEngine, SessionEngineIngress, SessionTransportEvent,
};
use assistant::{
    AssistantDeltaStream, AssistantTypewriter, assistant_delta_event, assistant_delta_parts,
    assistant_stream_end,
};
#[path = "runtime/assistant.rs"]
mod assistant;
use branch_poller::BranchPoller;
#[path = "runtime/branch_poller.rs"]
mod branch_poller;
#[path = "runtime/model_catalog.rs"]
mod model_catalog;
pub(crate) use model_catalog::{AvailableExpert, AvailableModel};
#[path = "runtime/support.rs"]
mod support;
use support::{
    ClipboardPasteChoice, ClipboardPasteContext, TERMINAL_TITLE_TICKS_PER_FRAME,
    choose_clipboard_paste, clipboard_image_attachments, format_terminal_title,
    mcp_discovery_description, next_attachment_id, next_submission_id, session_title_from_records,
};
#[path = "runtime/command_dispatch.rs"]
mod command_dispatch;
#[path = "runtime/history_tree_dialog.rs"]
mod history_tree_dialog;
#[path = "runtime/lifecycle.rs"]
mod lifecycle;
#[path = "runtime/permission_lifecycle.rs"]
mod permission_lifecycle;
#[path = "runtime/queued_prompt.rs"]
mod queued_prompt;
#[cfg(test)]
#[path = "runtime/session_cleanup.rs"]
mod session_cleanup;
#[path = "runtime/session_command_adapter.rs"]
mod session_command_adapter;
#[path = "runtime/session_dialog.rs"]
mod session_dialog;
use history_tree_dialog::history_tree_dialog_items;
use lifecycle::{active_turn_state, has_active_or_pending_session_turn};
use permission_lifecycle::PermissionLifecycleController;
use queued_prompt::{QueuedPromptDoneDisposition, QueuedPromptLifecycle};
use session_dialog::session_dialog_items;
#[cfg(test)]
use std::sync::Mutex as StdMutex;

const PAGE_SCROLL_ROWS: usize = 10;
const CONFIG_CUSTOM_CHOICE: &str = "custom";
const SESSION_ENGINE_UNAVAILABLE_MESSAGE: &str = "Session engine is no longer available";
// ~3 seconds at the 33ms TUI frame interval, long enough for deliberate chords.
const CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS: u8 = 90;
const TUI_FRAME_POLL_INTERVAL: Duration = Duration::from_millis(33);
const MAX_SESSION_EVENTS_PER_FRAME: usize = 256;
const MAX_SESSION_EVENT_TIME_PER_FRAME: Duration = Duration::from_millis(4);
const MAX_INPUT_EVENTS_PER_FRAME: usize = 64;

struct SessionEventBudget {
    started_at: Instant,
    consumed: usize,
    received: usize,
}

impl SessionEventBudget {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            consumed: 0,
            received: 0,
        }
    }

    fn can_receive(&self) -> bool {
        self.received < MAX_SESSION_EVENTS_PER_FRAME
            && self.started_at.elapsed() < MAX_SESSION_EVENT_TIME_PER_FRAME
    }

    fn receive(&mut self) {
        self.received = self.received.saturating_add(1);
    }

    fn can_process(&self) -> bool {
        self.consumed < MAX_SESSION_EVENTS_PER_FRAME
            && self.started_at.elapsed() < MAX_SESSION_EVENT_TIME_PER_FRAME
    }

    fn consume(&mut self) {
        self.consumed = self.consumed.saturating_add(1);
    }
}

/// Compatibility alias: session commands are owned by the backend boundary.
pub type RuntimeCommand = crate::session::SessionCommand;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupToast {
    message: String,
    kind: ToastKind,
}

impl StartupToast {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ToastKind::Error,
        }
    }

    #[cfg(test)]
    pub fn message(&self) -> &str {
        &self.message
    }

    #[cfg(test)]
    pub fn kind(&self) -> ToastKind {
        self.kind
    }
}

fn child_navigation_anchor(state: &TuiState) -> Option<String> {
    state
        .child_view_metadata()
        .map(|metadata| metadata.child_session_id)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SubmittedCommand {
    LocalOnly,
    Runtime(RuntimeCommand),
}

#[derive(Debug, Clone)]
struct ComposerDraft {
    input_buffer: String,
    input_cursor: usize,
    tokens: Vec<crate::tui::state::ComposerToken>,
}

pub trait RuntimeDrawer {
    fn draw(&mut self, state: &mut TuiState) -> io::Result<()>;
}

#[derive(Debug, Clone)]
struct OutputRateSample {
    child_session_id: Option<String>,
    started_at: Instant,
    streamed_bytes: u64,
    displayed_rate: Option<f64>,
    last_display_at: Option<Instant>,
}

pub struct TuiRuntime {
    state: TuiState,
    session_transport_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
    permission_lifecycle: PermissionLifecycleController,
    pending_question_handle: Option<RunnerQuestionRequest>,
    pending_question_child_session_id: Option<String>,
    interrupt_confirmation_pending: bool,
    submitted_prompts: Vec<String>,
    submitted_prompt_drafts: Vec<ComposerDraft>,
    queued_prompts: VecDeque<UserMessageSubmission>,
    queued_prompt_lifecycle: QueuedPromptLifecycle,
    session_turn_active: bool,
    last_output_rate_graph_sample_at: Option<Instant>,
    session_resume_pending: bool,
    /// Background `/resume` directory scan; polled each frame so the UI never blocks.
    session_list_rx: Option<mpsc::UnboundedReceiver<anyhow::Result<Vec<SessionSummary>>>>,
    /// One-shot background release check. Failures are logged and never interrupt the TUI.
    update_check_rx: Option<mpsc::UnboundedReceiver<anyhow::Result<Option<String>>>>,
    /// One-shot background archive pass started at TUI startup. Its report only
    /// reaches the UI for sessions that exhausted their archive attempts.
    archive_pass_rx: Option<mpsc::UnboundedReceiver<anyhow::Result<ArchiveRunReport>>>,
    current_turn_output_tokens: u64,
    output_rate_samples: Vec<OutputRateSample>,
    history_selection: Option<usize>,
    history_draft: Option<ComposerDraft>,
    available_models: Vec<AvailableModel>,
    available_experts: Vec<AvailableExpert>,
    branch_poller: BranchPoller,
    sessions_dir: PathBuf,
    workspace_key: Option<String>,
    session_summaries: Vec<SessionSummary>,
    preferences_dir: PathBuf,
    assistant_typewriters: Vec<AssistantTypewriter>,
    deferred_session_events: VecDeque<SessionTransportEvent>,
    session_transport_stream_closed: bool,
    session_transport_stream_close_reported: bool,
    session_title: Option<String>,
    spinner_frame: usize,
    theme_preview_original: Option<(String, Option<Theme>)>,
    config_path: Option<PathBuf>,
    config_draft: Option<toml_edit::DocumentMut>,
}

impl TuiRuntime {
    pub fn new(
        state: TuiState,
        session_transport_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
        available_models: Vec<AvailableModel>,
        available_experts: Vec<AvailableExpert>,
        sessions_dir: PathBuf,
        preferences_dir: PathBuf,
    ) -> Self {
        Self {
            state,
            session_transport_rx,
            permission_lifecycle: PermissionLifecycleController::default(),
            pending_question_handle: None,
            pending_question_child_session_id: None,
            interrupt_confirmation_pending: false,
            submitted_prompts: Vec::new(),
            submitted_prompt_drafts: Vec::new(),
            queued_prompts: VecDeque::new(),
            queued_prompt_lifecycle: QueuedPromptLifecycle::default(),
            session_turn_active: false,
            last_output_rate_graph_sample_at: None,
            session_resume_pending: false,
            session_list_rx: None,
            update_check_rx: None,
            archive_pass_rx: None,
            current_turn_output_tokens: 0,
            output_rate_samples: Vec::new(),
            history_selection: None,
            history_draft: None,
            available_models,
            available_experts,
            branch_poller: BranchPoller::new(),
            sessions_dir,
            workspace_key: None,
            session_summaries: Vec::new(),
            preferences_dir,
            assistant_typewriters: Vec::new(),
            deferred_session_events: VecDeque::new(),
            session_transport_stream_closed: false,
            session_transport_stream_close_reported: false,
            session_title: None,
            spinner_frame: 0,
            theme_preview_original: None,
            config_path: None,
            config_draft: None,
        }
    }

    pub fn set_workspace_dir(&mut self, workspace_dir: PathBuf) {
        self.workspace_key = Some(crate::transcript::workspace_root(&workspace_dir));
        self.branch_poller.set_workspace_dir(workspace_dir);
        self.poll_git_branch();
    }

    pub fn set_config_path(&mut self, config_path: PathBuf) {
        self.config_path = Some(config_path);
    }

    fn start_update_check(&mut self) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.update_check_rx = Some(rx);
        std::thread::spawn(move || {
            let result = crate::updater::available_update()
                .map(|update| update.map(|update| update.latest_version));
            let _ = tx.send(result);
        });
    }

    /// Archive idle session families once, off the frame loop. The pass is
    /// silent while it succeeds, and never runs from one-shot or JSON CLI modes
    /// because those never enter the TUI.
    fn start_session_archive_pass(&mut self) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.archive_pass_rx = Some(rx);
        let sessions_dir = self.sessions_dir.clone();
        std::thread::spawn(move || {
            // The pass reads the settings from the same config file the process
            // started from, without holding a live AppConfig in the TUI.
            let config = match crate::config::AppConfig::load() {
                Ok(config) => config.global.session_archive,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "skipping the session archive pass; the config is unreadable"
                    );
                    return;
                }
            };
            let _ = tx.send(archive::run_archive_pass(
                &sessions_dir,
                config,
                ArchiveBudget::default(),
            ));
        });
    }

    pub fn state(&self) -> &TuiState {
        &self.state
    }

    #[cfg(test)]
    pub fn state_mut(&mut self) -> &mut TuiState {
        &mut self.state
    }

    #[cfg(test)]
    pub fn available_models(&self) -> &[AvailableModel] {
        &self.available_models
    }

    fn terminal_title(&self) -> String {
        // 等待用户回答的提示占住 spinner 位并用标记字符替代：提问是 `?`，审批是 `!`。
        let pending_marker = if self.state.pending_question.is_some() {
            Some('?')
        } else if self.state.pending_permission.is_some() {
            Some('!')
        } else {
            None
        };
        let title = format_terminal_title(
            self.session_title.as_deref(),
            (pending_marker.is_none() && self.has_active_or_pending_session_turn())
                .then_some(self.spinner_frame / TERMINAL_TITLE_TICKS_PER_FRAME),
        );
        match pending_marker {
            Some(marker) => format!("{marker} {title}"),
            None => title,
        }
    }

    fn update_terminal_title(&self, terminal: &mut OwnedTerminal) -> io::Result<()> {
        terminal.set_title(&self.terminal_title())
    }

    fn show_toast(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.state.show_toast(message, kind);
    }

    #[cfg(test)]
    pub fn submitted_prompts(&self) -> &[String] {
        &self.submitted_prompts
    }

    #[cfg(test)]
    pub fn pending_permission_handle(&self) -> Option<&RunnerPermissionRequest> {
        self.permission_lifecycle.handle()
    }

    fn begin_pending_question(
        &mut self,
        request: crate::tool::QuestionRequest,
        handle: RunnerQuestionRequest,
        child_session_id: Option<String>,
    ) -> Result<()> {
        if self.state.pending_question.is_some() || self.permission_lifecycle.is_pending() {
            return Err(anyhow!("interactive request already pending"));
        }

        let origin_label = child_session_id
            .as_ref()
            .map(|_| "Child question".to_string());
        self.state.pending_question = Some(PendingQuestionState::new(request, origin_label));
        self.pending_question_handle = Some(handle);
        self.pending_question_child_session_id = child_session_id;
        self.state.phase = super::state::AppPhase::WaitingForPermission;
        self.state.toast = None;
        Ok(())
    }

    fn clear_pending_question(&mut self) {
        self.state.pending_question = None;
        self.pending_question_handle = None;
        self.pending_question_child_session_id = None;
        if matches!(
            self.state.phase,
            super::state::AppPhase::WaitingForPermission
        ) {
            self.state.phase = super::state::AppPhase::Running;
        }
        self.state.sync_input_phase();
    }

    fn is_stale_question_interaction(error: &anyhow::Error) -> bool {
        matches!(
            error.to_string().as_str(),
            "question response receiver dropped" | "question request already resolved"
        )
    }

    fn cancel_pending_question(&mut self, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let handle = self.pending_question_handle.take();
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.cancel(reason.clone())
        {
            if Self::is_stale_question_interaction(&error) {
                tracing::warn!(error = %error, "ignored stale question cancellation");
            } else {
                return Err(error);
            }
        }
        Ok(())
    }

    fn cancel_pending_question_if_parent(&mut self, reason: &str) {
        if self.pending_question_child_session_id.is_none()
            && (self.state.pending_question.is_some() || self.pending_question_handle.is_some())
        {
            let _ = self.cancel_pending_question(reason);
        }
    }

    fn submit_pending_question(&mut self) -> Result<()> {
        let unanswered_tab = self
            .state
            .pending_question
            .as_ref()
            .and_then(PendingQuestionState::first_unanswered_tab);

        if let Some(tab_index) = unanswered_tab {
            if let Some(question) = self.state.pending_question.as_mut() {
                question.focus_tab(tab_index);
            }
            self.state.show_toast(
                self.state.t("runtime.answer_all_questions"),
                ToastKind::Info,
            );
            return Ok(());
        }

        let Some(question) = self.state.pending_question.as_ref() else {
            self.state
                .show_toast(self.state.t("runtime.no_question_pending"), ToastKind::Info);
            return Ok(());
        };
        if question.has_invalid_single_response() {
            self.state
                .show_toast(self.state.t("runtime.single_select_only"), ToastKind::Info);
            return Ok(());
        }

        let response = question.build_response();
        let handle = self.pending_question_handle.take();
        self.state.toast = None;
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.answer(response)
        {
            if Self::is_stale_question_interaction(&error) {
                tracing::warn!(error = %error, "ignored stale question answer");
            } else {
                return Err(error);
            }
            return Ok(());
        }
        Ok(())
    }

    fn insert_pending_question_text(&mut self, text: &str) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .filter(|question| question.editing_custom)
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.insert_custom_edit(text);
        }
    }

    fn backspace_pending_question_text(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.backspace_custom_edit();
        }
    }

    fn delete_pending_question_text(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.delete_custom_edit();
        }
    }

    fn move_pending_question_cursor_left(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_left();
        }
    }

    fn move_pending_question_cursor_right(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_right();
        }
    }

    fn move_pending_question_cursor_home(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_home();
        }
    }

    fn move_pending_question_cursor_end(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.move_custom_cursor_end();
        }
    }
}

// ── 会话事件归约与后台轮询 ─────────────────────
fn enqueue_merged_event(queue: &mut VecDeque<SessionTransportEvent>, event: SessionTransportEvent) {
    if let Some(previous) = queue.back_mut()
        && try_merge_adjacent_transport_events(previous, &event)
    {
        return;
    }
    queue.push_back(event);
}

fn is_transcript_view_projection(event: &SessionTransportEvent) -> bool {
    matches!(
        event,
        SessionTransportEvent::ChildSessionViewed { .. }
            | SessionTransportEvent::ParentSessionViewed { .. }
    )
}

fn try_merge_adjacent_transport_events(
    previous: &mut SessionTransportEvent,
    next: &SessionTransportEvent,
) -> bool {
    if let (
        SessionTransportEvent::AssistantDelta(previous),
        SessionTransportEvent::AssistantDelta(next),
    ) = (&mut *previous, next)
        && previous.message_id == next.message_id
    {
        previous.delta.push_str(&next.delta);
        return true;
    }
    if let (
        SessionTransportEvent::ReasoningDelta(previous),
        SessionTransportEvent::ReasoningDelta(next),
    ) = (&mut *previous, next)
        && previous.item_id == next.item_id
    {
        previous.delta.push_str(&next.delta);
        return true;
    }
    if let (
        SessionTransportEvent::ToolOutputDelta(previous),
        SessionTransportEvent::ToolOutputDelta(next),
    ) = (&mut *previous, next)
        && previous.call_id == next.call_id
        && previous.stream == next.stream
    {
        previous.chunk.push_str(&next.chunk);
        return true;
    }
    if let (
        SessionTransportEvent::ChildSessionEvent {
            child_session_id: previous_child,
            agent_name: previous_agent,
            parent_tool_call_id: previous_parent,
            event: previous_event,
        },
        SessionTransportEvent::ChildSessionEvent {
            child_session_id: next_child,
            agent_name: next_agent,
            parent_tool_call_id: next_parent,
            event: next_event,
        },
    ) = (&mut *previous, next)
        && previous_child == next_child
        && previous_agent == next_agent
        && previous_parent == next_parent
        && try_merge_adjacent_session_events(previous_event, next_event)
    {
        return true;
    }
    false
}

fn try_merge_adjacent_session_events(previous: &mut SessionEvent, next: &SessionEvent) -> bool {
    if let (SessionEvent::AssistantDelta(previous), SessionEvent::AssistantDelta(next)) =
        (&mut *previous, next)
        && previous.message_id == next.message_id
    {
        previous.delta.push_str(&next.delta);
        return true;
    }
    if let (SessionEvent::ReasoningDelta(previous), SessionEvent::ReasoningDelta(next)) =
        (&mut *previous, next)
        && previous.item_id == next.item_id
    {
        previous.delta.push_str(&next.delta);
        return true;
    }
    if let (SessionEvent::ToolOutputDelta(previous), SessionEvent::ToolOutputDelta(next)) =
        (&mut *previous, next)
        && previous.call_id == next.call_id
        && previous.stream == next.stream
    {
        previous.chunk.push_str(&next.chunk);
        return true;
    }
    false
}

fn process_terminal_event(
    runtime: &mut TuiRuntime,
    event: Event,
    ingress: &SessionEngineIngress,
) -> Result<()> {
    match event {
        Event::Key(key) => {
            // Windows may report key-up records as key events; only handle presses.
            if key.kind != KeyEventKind::Press {
                return Ok(());
            }
            let action = map_key_event(runtime.state(), key);
            if let Some(command) = runtime.handle_input_action(action)? {
                command_dispatch::dispatch_command(runtime, command, ingress, true);
            }
        }
        Event::Mouse(mouse) => {
            let action = map_mouse_event(runtime.state(), mouse);
            if let Some(command) = runtime.handle_input_action(action)? {
                command_dispatch::dispatch_command(runtime, command, ingress, false);
            }
        }
        Event::Paste(text) => {
            let action = map_paste_event(runtime.state(), text);
            if let Some(command) = runtime.handle_input_action(action)? {
                command_dispatch::dispatch_command(runtime, command, ingress, false);
            }
        }
        Event::Resize(_, _) | Event::FocusGained | Event::FocusLost => {}
    }
    Ok(())
}

impl TuiRuntime {
    pub fn try_drain_session_events(&mut self) {
        // Leave both time and event slots in every frame for terminal input.
        // An unbounded model stream must not delay a confirmed Esc indefinitely.
        let mut budget = SessionEventBudget::new();
        self.advance_assistant_typewriters_with_budget(Instant::now(), &mut budget);
        self.poll_session_list();
        self.poll_update_check();
        self.poll_session_archive_pass();
        self.poll_git_branch();

        let mut batch = VecDeque::new();
        let mut stream_closed = false;
        while budget.can_receive() {
            match self.session_transport_rx.try_recv() {
                Ok(event) => {
                    budget.receive();
                    self.observe_output_rate_transport_event(&event, Instant::now());
                    enqueue_merged_event(&mut batch, event);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    stream_closed = true;
                    break;
                }
            }
        }
        while budget.can_process() {
            let Some(event) = batch.pop_front() else {
                break;
            };
            budget.consume();
            self.consume_observed_session_transport_event(event);
        }
        // Preserve events that were received but could not be applied within this
        // frame. They remain ordered after already-deferred work and are handled by
        // the same budget on the next frame.
        while let Some(event) = batch.pop_front() {
            self.enqueue_deferred_session_event(event);
        }
        if stream_closed {
            self.session_transport_stream_closed = true;
            self.flush_session_events();
        }
        if self.session_transport_stream_closed
            && !self.session_transport_stream_close_reported
            && self.assistant_typewriters.is_empty()
            && self.deferred_session_events.is_empty()
        {
            self.handle_session_event_stream_closed();
        }
    }

    fn poll_git_branch(&mut self) {
        self.branch_poller.poll(&mut self.state);
    }

    fn poll_update_check(&mut self) {
        let Some(rx) = self.update_check_rx.as_mut() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(Some(version))) => {
                self.update_check_rx = None;
                let message = self
                    .state
                    .t_fmt("runtime.update_available", &[("version", version.as_str())]);
                self.state.show_toast(message, ToastKind::Info);
            }
            Ok(Ok(None)) => self.update_check_rx = None,
            Ok(Err(error)) => {
                self.update_check_rx = None;
                tracing::debug!(%error, "background update check failed");
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
            Err(mpsc::error::TryRecvError::Disconnected) => self.update_check_rx = None,
        }
    }

    fn poll_session_archive_pass(&mut self) {
        let Some(rx) = self.archive_pass_rx.as_mut() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(report)) => {
                self.archive_pass_rx = None;
                if let Some((first, rest)) = report.anomalies.split_first() {
                    let message = if rest.is_empty() {
                        self.state
                            .t_fmt("runtime.session_archive_anomaly", &[("session", first)])
                    } else {
                        let count = rest.len().to_string();
                        self.state.t_fmt(
                            "runtime.session_archive_anomaly_more",
                            &[("session", first), ("count", &count)],
                        )
                    };
                    self.state.show_toast(message, ToastKind::Error);
                }
            }
            Ok(Err(error)) => {
                self.archive_pass_rx = None;
                tracing::warn!(%error, "session archive pass failed");
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
            Err(mpsc::error::TryRecvError::Disconnected) => self.archive_pass_rx = None,
        }
    }

    fn poll_session_list(&mut self) {
        let Some(rx) = self.session_list_rx.as_mut() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(sessions)) => {
                self.session_list_rx = None;
                if sessions.is_empty() {
                    self.push_command_notice(self.state.t("runtime.no_sessions_found"));
                    return;
                }
                let mut dialog = DialogState::new(
                    DialogKind::SessionPicker,
                    self.state.t("ui.sessions_section"),
                    None,
                    Vec::new(),
                );
                let items = self.session_rows(&sessions, dialog.session_scope);
                dialog.replace_items(items);
                self.session_summaries = sessions;
                self.state.open_dialog(dialog);
            }
            Ok(Err(error)) => {
                self.session_list_rx = None;
                self.state.show_toast(
                    self.state.t_fmt(
                        "runtime.failed_list_sessions",
                        &[("error", &error.to_string())],
                    ),
                    ToastKind::Error,
                );
            }
            Err(mpsc::error::TryRecvError::Empty) => {}
            Err(mpsc::error::TryRecvError::Disconnected) => {
                self.session_list_rx = None;
                self.state.show_toast(
                    self.state.t("runtime.session_list_worker_stopped"),
                    ToastKind::Error,
                );
            }
        }
    }

    fn handle_session_event_stream_closed(&mut self) {
        self.session_transport_stream_close_reported = true;
        let terminalize = self.has_active_or_pending_session_turn() || self.session_resume_pending;
        if !self.state.quit_requested {
            self.permission_lifecycle.clear();
            let _ = self
                .cancel_pending_question("question cancelled because the session engine stopped");
            self.state.mark_child_sessions_disconnected();
            self.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
                "TUI session event stream closed unexpectedly",
            )));
            if terminalize {
                self.apply_session_transport_event(SessionTransportEvent::Done);
            }
            self.clear_unaccepted_queued_prompt();
        }
    }

    #[cfg(test)]
    fn consume_session_transport_event(&mut self, event: SessionTransportEvent) {
        self.observe_output_rate_transport_event(&event, Instant::now());
        self.consume_observed_session_transport_event(event);
    }

    fn consume_observed_session_transport_event(&mut self, event: SessionTransportEvent) {
        // Paced text bypasses the queue unless a queued event precedes it.
        if let Some((stream, agent_name, delta)) = assistant_delta_parts(&event) {
            if self.deferred_session_events.is_empty() {
                self.push_stream_delta(stream, agent_name, delta);
            } else {
                self.enqueue_deferred_session_event(event);
            }
            return;
        }

        // View projections change which timeline owns subsequent paint work, so
        // commit any paced text before applying the navigation snapshot.
        if is_transcript_view_projection(&event) {
            if self.deferred_session_events.is_empty() {
                self.end_pacing();
                self.apply_session_transport_event(event);
            } else {
                self.enqueue_deferred_session_event(event);
            }
            return;
        }

        if self.has_pending_text() || !self.deferred_session_events.is_empty() {
            self.enqueue_deferred_session_event(event);
        } else {
            self.apply_session_transport_event(event);
        }
    }

    fn enqueue_deferred_session_event(&mut self, event: SessionTransportEvent) {
        enqueue_merged_event(&mut self.deferred_session_events, event);
    }

    #[cfg(test)]
    fn advance_assistant_typewriter_by(&mut self, elapsed: Duration) {
        let now = self
            .assistant_typewriters
            .first()
            .map(|typewriter| typewriter.last_frame_at + elapsed)
            .unwrap_or_else(Instant::now);
        let mut budget = SessionEventBudget {
            started_at: Instant::now(),
            consumed: 0,
            received: 0,
        };
        self.advance_assistant_typewriters_with_budget(now, &mut budget);
    }

    fn advance_assistant_typewriters_with_budget(
        &mut self,
        now: Instant,
        budget: &mut SessionEventBudget,
    ) {
        let view_projection_pending = self
            .deferred_session_events
            .iter()
            .any(is_transcript_view_projection);
        let mut index = 0;
        while index < self.assistant_typewriters.len() {
            if !budget.can_process() {
                break;
            }
            let (stream, agent_name, released) = {
                let typewriter = &mut self.assistant_typewriters[index];
                let released = if view_projection_pending {
                    typewriter.take_pending()
                } else {
                    typewriter.take_frame(now)
                };
                (
                    typewriter.stream.clone(),
                    typewriter.agent_name.clone(),
                    released,
                )
            };
            if !released.is_empty() {
                budget.consume();
                self.apply_session_transport_event(assistant_delta_event(
                    &stream,
                    &agent_name,
                    released,
                ));
            }
            index += 1;
        }
        self.apply_deferred_session_events_with_budget(budget);
    }

    fn typewriter_index(&self, stream: &AssistantDeltaStream) -> Option<usize> {
        self.assistant_typewriters
            .iter()
            .position(|typewriter| &typewriter.stream == stream)
    }

    fn push_stream_delta(
        &mut self,
        stream: AssistantDeltaStream,
        agent_name: Option<String>,
        delta: String,
    ) {
        let now = Instant::now();
        let index = self.typewriter_index(&stream).unwrap_or_else(|| {
            self.assistant_typewriters
                .push(AssistantTypewriter::new(stream, agent_name, now));
            self.assistant_typewriters.len() - 1
        });
        self.assistant_typewriters[index].push(&delta, now);
    }

    fn has_pending_text(&self) -> bool {
        self.assistant_typewriters
            .iter()
            .any(|typewriter| !typewriter.pending_text().is_empty())
    }

    /// Commit what every stream has already produced, without ending its pacing.
    fn commit_pending_text(&mut self) {
        for (stream, agent_name, text) in self.take_pending_text() {
            self.apply_session_transport_event(assistant_delta_event(&stream, &agent_name, text));
        }
    }

    /// Commit the paced text of every stream and stop pacing them.
    fn end_pacing(&mut self) {
        let pending = self.take_pending_text();
        self.assistant_typewriters.clear();
        for (stream, agent_name, text) in pending {
            self.apply_session_transport_event(assistant_delta_event(&stream, &agent_name, text));
        }
    }

    /// Finish everything a closed transport still holds, so the drain is bounded.
    fn flush_session_events(&mut self) {
        self.end_pacing();
        while let Some(event) = self.deferred_session_events.pop_front() {
            self.apply_session_transport_event(event);
        }
    }

    fn take_pending_text(&mut self) -> Vec<(AssistantDeltaStream, Option<String>, String)> {
        self.assistant_typewriters
            .iter_mut()
            .filter_map(|typewriter| {
                let text = typewriter.take_pending();
                (!text.is_empty()).then(|| {
                    (
                        typewriter.stream.clone(),
                        typewriter.agent_name.clone(),
                        text,
                    )
                })
            })
            .collect()
    }

    /// Queued events keep their arrival order and never wait for paced text.
    fn apply_deferred_session_events_with_budget(&mut self, budget: &mut SessionEventBudget) {
        while !self.deferred_session_events.is_empty() && budget.can_process() {
            let event = self
                .deferred_session_events
                .pop_front()
                .expect("deferred queue checked above");
            budget.consume();
            if let Some((stream, agent_name, delta)) = assistant_delta_parts(&event) {
                self.push_stream_delta(stream, agent_name, delta);
                continue;
            }
            self.commit_pending_text();
            self.apply_session_transport_event(event.clone());
            if let Some(stream) = assistant_stream_end(&event) {
                self.end_stream_pacing(&stream);
            }
        }
    }

    fn end_stream_pacing(&mut self, stream: &AssistantDeltaStream) {
        let Some(index) = self.typewriter_index(stream) else {
            return;
        };
        let typewriter = &mut self.assistant_typewriters[index];
        let text = typewriter.take_pending();
        let (stream, agent_name) = (typewriter.stream.clone(), typewriter.agent_name.clone());
        self.assistant_typewriters.remove(index);
        if !text.is_empty() {
            self.apply_session_transport_event(assistant_delta_event(&stream, &agent_name, text));
        }
    }

    fn observe_output_rate_transport_event(&mut self, event: &SessionTransportEvent, now: Instant) {
        match event {
            SessionTransportEvent::UserMessage(_) => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.is_some());
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::UserMessage(_),
                ..
            } => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.as_deref() != Some(child_session_id));
            }
            SessionTransportEvent::AssistantDelta(delta) => {
                self.update_output_rate_sample(None, &delta.delta, now);
            }
            SessionTransportEvent::ReasoningDelta(delta) => {
                self.update_output_rate_sample(None, &delta.delta, now);
            }
            SessionTransportEvent::ToolPending(tool) => {
                self.update_output_rate_sample(None, &tool.name, now);
            }
            SessionTransportEvent::ToolStarted(tool) => {
                if let Some(arguments) = tool.arguments.as_deref() {
                    self.update_existing_output_rate_sample(None, arguments, now);
                }
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::AssistantDelta(delta),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &delta.delta, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ReasoningDelta(delta),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &delta.delta, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ToolPending(tool),
                ..
            } => {
                self.update_output_rate_sample(Some(child_session_id), &tool.name, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::ToolStarted(tool),
                ..
            } => {
                if let Some(arguments) = tool.arguments.as_deref() {
                    self.update_existing_output_rate_sample(Some(child_session_id), arguments, now);
                }
            }
            _ => self.finish_output_rate_for_transport_event(event, now),
        }
    }

    fn sample_output_rate_graph(&mut self, now: Instant) {
        const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);
        if self
            .last_output_rate_graph_sample_at
            .is_some_and(|last| now.saturating_duration_since(last) < SAMPLE_INTERVAL)
        {
            return;
        }
        self.last_output_rate_graph_sample_at = Some(now);
        let rate = if self.session_turn_active {
            self.state.active_output_token_rate()
        } else {
            None
        };
        self.state.push_output_rate_graph_sample(rate);
    }

    fn set_output_token_rate_for_session(
        &mut self,
        child_session_id: Option<&str>,
        rate: Option<u64>,
    ) {
        if let Some(child_session_id) = child_session_id {
            self.state
                .set_child_output_token_rate(child_session_id, rate);
        } else {
            self.state.set_output_token_rate(rate);
        }
    }

    fn update_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        delta: &str,
        now: Instant,
    ) {
        let sample_index = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
            .unwrap_or_else(|| {
                let displayed_rate = if let Some(child_session_id) = child_session_id {
                    self.state
                        .child_output_token_rate(child_session_id)
                        .map(|rate| rate as f64)
                } else {
                    self.state.output_token_rate.map(|rate| rate as f64)
                };
                self.output_rate_samples.push(OutputRateSample {
                    child_session_id: child_session_id.map(str::to_owned),
                    started_at: now,
                    streamed_bytes: 0,
                    displayed_rate,
                    last_display_at: None,
                });
                self.output_rate_samples.len() - 1
            });
        self.update_output_rate_sample_at(sample_index, delta, now);
    }

    fn update_existing_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        delta: &str,
        now: Instant,
    ) {
        if let Some(sample_index) = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
        {
            self.update_output_rate_sample_at(sample_index, delta, now);
        }
    }

    fn update_output_rate_sample_at(&mut self, sample_index: usize, delta: &str, now: Instant) {
        const MIN_SAMPLE_DURATION: Duration = Duration::from_millis(500);
        const MIN_DISPLAY_INTERVAL: Duration = Duration::from_millis(250);
        const RATE_SMOOTHING: f64 = 0.25;
        const MAX_RATE_INCREASE_FACTOR: f64 = 2.0;

        let sample = &mut self.output_rate_samples[sample_index];
        sample.streamed_bytes = sample.streamed_bytes.saturating_add(delta.len() as u64);
        let elapsed = now.saturating_duration_since(sample.started_at);
        if elapsed < MIN_SAMPLE_DURATION
            || sample.last_display_at.is_some_and(|last_display_at| {
                now.saturating_duration_since(last_display_at) < MIN_DISPLAY_INTERVAL
            })
        {
            return;
        }
        let estimated_tokens = sample.streamed_bytes.div_ceil(3);
        let instantaneous_rate = estimated_tokens as f64 / elapsed.as_secs_f64();
        let displayed_rate = match sample.displayed_rate {
            Some(previous) => {
                let smoothed =
                    previous * (1.0 - RATE_SMOOTHING) + instantaneous_rate * RATE_SMOOTHING;
                smoothed.min(previous * MAX_RATE_INCREASE_FACTOR)
            }
            None => instantaneous_rate,
        };
        sample.displayed_rate = Some(displayed_rate);
        sample.last_display_at = Some(now);
        let child_session_id = sample.child_session_id.clone();
        self.set_output_token_rate_for_session(
            child_session_id.as_deref(),
            Some(displayed_rate.round() as u64),
        );
    }

    fn finish_output_rate_for_transport_event(
        &mut self,
        event: &SessionTransportEvent,
        now: Instant,
    ) {
        match event {
            SessionTransportEvent::TokenUsage(usage) if usage.output_tokens > 0 => {
                self.finish_output_rate_sample(None, usage.output_tokens, now);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                event: SessionEvent::TokenUsage(usage),
                ..
            } if usage.output_tokens > 0 => {
                self.finish_output_rate_sample(Some(child_session_id), usage.output_tokens, now);
            }
            _ => {}
        }
    }

    fn finish_output_rate_sample(
        &mut self,
        child_session_id: Option<&str>,
        output_tokens: u64,
        now: Instant,
    ) {
        let Some(sample_index) = self
            .output_rate_samples
            .iter()
            .position(|sample| sample.child_session_id.as_deref() == child_session_id)
        else {
            return;
        };
        let sample = self.output_rate_samples.remove(sample_index);
        let elapsed = now.saturating_duration_since(sample.started_at);
        if elapsed.is_zero() {
            return;
        }
        let rate = (output_tokens as f64 / elapsed.as_secs_f64()).round() as u64;
        self.set_output_token_rate_for_session(sample.child_session_id.as_deref(), Some(rate));
    }

    pub fn apply_session_transport_event(&mut self, event: SessionTransportEvent) {
        let mut suppress_session_event = false;

        match &event {
            SessionTransportEvent::QuestionRequested { request, handle }
                if self
                    .begin_pending_question(request.clone(), handle.clone(), None)
                    .is_err() =>
            {
                let _ = handle.cancel("another interactive request is already pending");
                self.state.show_toast(
                    self.state.t("runtime.question_already_pending"),
                    ToastKind::Info,
                );
                suppress_session_event = true;
            }
            SessionTransportEvent::PermissionRequested { event, handle } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state.show_toast(
                        self.state.t("runtime.question_already_pending"),
                        ToastKind::Info,
                    );
                    suppress_session_event = true;
                } else if let Err(handle) = self
                    .permission_lifecycle
                    .begin_parent(event.clone(), handle.clone())
                {
                    let _ = handle.deny();
                    self.state
                        .show_toast(self.state.t("runtime.permission_pending"), ToastKind::Info);
                    suppress_session_event = true;
                } else {
                    self.state.toast = None;
                }
            }
            SessionTransportEvent::ChildQuestionRequested {
                child_session_id,
                request,
                handle,
            } if self
                .begin_pending_question(
                    request.clone(),
                    handle.clone(),
                    Some(child_session_id.clone()),
                )
                .is_err() =>
            {
                let _ = handle.cancel("another interactive request is already pending");
                self.state.show_toast(
                    self.state.t("runtime.question_already_pending"),
                    ToastKind::Info,
                );
                suppress_session_event = true;
            }
            SessionTransportEvent::ChildPermissionRequested {
                child_session_id,
                agent_name,
                parent_tool_call_id,
                event,
                handle,
            } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state.show_toast(
                        self.state.t("runtime.question_already_pending"),
                        ToastKind::Info,
                    );
                } else if let Err(handle) = self.permission_lifecycle.begin_child(
                    child_session_id.clone(),
                    event.clone(),
                    handle.clone(),
                ) {
                    let _ = handle.deny();
                    self.state
                        .show_toast(self.state.t("runtime.permission_pending"), ToastKind::Info);
                } else {
                    self.state.toast = None;
                    self.state.apply_child_session_event_with_agent(
                        child_session_id,
                        agent_name.as_deref(),
                        parent_tool_call_id.as_deref(),
                        SessionEvent::PermissionRequested(event.clone()),
                    );
                }
            }
            SessionTransportEvent::HistorianStatus {
                session_id,
                running,
                failed,
            } if self.state.session_id.as_deref() == Some(session_id.as_str()) => {
                self.state.historian_status = Some((session_id.clone(), *running, *failed));
            }
            SessionTransportEvent::PermissionResolved(resolution)
                if self.pending_permission_matches_call(&resolution.call_id, None) =>
            {
                self.permission_lifecycle.clear();
            }
            SessionTransportEvent::Done => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.is_some());
                self.permission_lifecycle.clear_if_parent();
                self.cancel_pending_question_if_parent("question cancelled because the turn ended");
                self.interrupt_confirmation_pending = false;
                match self.queued_prompt_lifecycle.done_disposition() {
                    QueuedPromptDoneDisposition::ReadyForNextDispatch => {
                        self.queued_prompt_lifecycle.mark_dispatch_ready();
                    }
                    QueuedPromptDoneDisposition::PreserveInFlight => {}
                    QueuedPromptDoneDisposition::ConsumeFailedAcceptedPrompt(prompt) => {
                        if self
                            .queued_prompts
                            .front()
                            .is_some_and(|queued| queued.id == prompt.id)
                        {
                            self.queued_prompts.pop_front();
                            self.state
                                .timeline
                                .remove_first_queued_user_message_preview(&prompt.id);
                        }
                        self.queued_prompt_lifecycle =
                            QueuedPromptLifecycle::idle(!self.queued_prompts.is_empty());
                    }
                }
                self.session_turn_active = false;
            }
            SessionTransportEvent::Error(_) => {
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.is_some());
                self.interrupt_confirmation_pending = false;
                self.session_resume_pending = false;
                self.queued_prompt_lifecycle.record_error();
            }
            SessionTransportEvent::FastModeChanged { enabled } => {
                self.state.set_fast_mode_enabled(*enabled);
                self.persist_fast_mode(*enabled);
            }
            SessionTransportEvent::ModelChanged { model_id } => {
                self.apply_restored_model(model_id.clone());
                self.state.set_provider_label_from_model_route(model_id);
                self.state.clear_pending_model_if(model_id);
            }
            SessionTransportEvent::ExpertAllowedModelsChanged {
                agent_name,
                model_ids,
            } => {
                if let Some(expert) = self
                    .available_experts
                    .iter_mut()
                    .find(|expert| expert.agent_name == *agent_name)
                {
                    expert.allowed_models = model_ids.clone();
                }
                if let Some(dialog) = self.state.dialog_mut().filter(|dialog| {
                    matches!(
                        &dialog.kind,
                        DialogKind::ExpertModelPicker(open_agent) if open_agent == agent_name
                    )
                }) {
                    for item in &mut dialog.items {
                        item.checked = model_ids.contains(&item.id);
                    }
                }
                self.refresh_open_agent_picker();
            }
            SessionTransportEvent::PermissionModeChanged { mode } => {
                self.state.set_permission_mode_label(mode.clone());
                self.state.clear_pending_permission_mode_if(mode);
            }
            SessionTransportEvent::ReasoningEffortChanged { effort } => {
                let label = reasoning_effort_status_label(Some(effort.clone()));
                self.state.set_reasoning_effort_label(Some(label.clone()));
                self.state.clear_pending_reasoning_effort_if(&label);
            }
            SessionTransportEvent::FakeClientChanged { client } => {
                self.state.set_fake_client(*client);
            }
            SessionTransportEvent::BackgroundSubagentCompleted {
                parent_tool_call_id,
                result,
            } => {
                self.state
                    .update_background_subagent_result(parent_tool_call_id.as_deref(), result);
            }
            SessionTransportEvent::SettingChangeFailed { command } => {
                self.clear_failed_pending_setting(command);
                suppress_session_event = true;
            }
            SessionTransportEvent::ModelCatalogUpdated(catalog) => {
                self.apply_model_catalog_update(catalog);
                suppress_session_event = true;
            }
            SessionTransportEvent::QueuedPromptAccepted { prompt } => {
                self.queued_prompt_lifecycle.accept(&prompt.id);
            }
            SessionTransportEvent::SessionTitleUpdated { session_id, title }
                if self.state.session_id.as_deref() == Some(session_id) =>
            {
                self.session_title = Some(title.clone());
            }
            SessionTransportEvent::Interrupted => {
                let interrupting_message = self.state.t("runtime.interrupting");
                if self
                    .state
                    .toast()
                    .is_some_and(|toast| toast.message == interrupting_message)
                {
                    self.state.toast = None;
                }
                self.output_rate_samples
                    .retain(|sample| sample.child_session_id.is_some());
                self.permission_lifecycle.clear_if_parent();
                let _ = self
                    .cancel_pending_question("question cancelled because the turn was interrupted");
                self.interrupt_confirmation_pending = false;
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.session_turn_active = false;
                self.state.activate_all_queued_user_message_previews();
            }
            SessionTransportEvent::UserMessage(user_message) => {
                self.queued_prompt_lifecycle.clear_dispatch_ready();
                self.session_turn_active = true;
                self.current_turn_output_tokens = 0;

                if self
                    .queued_prompt_lifecycle
                    .dispatched_submission_id()
                    .is_some_and(|dispatched| dispatched == user_message.submission_id)
                    && self
                        .queued_prompts
                        .front()
                        .is_some_and(|queued| queued.id == user_message.submission_id)
                {
                    self.queued_prompt_lifecycle
                        .resolve_user_message(&user_message.submission_id);
                    self.queued_prompts.pop_front();
                    suppress_session_event = self
                        .state
                        .activate_queued_user_message(&user_message.submission_id);
                } else if self.state.timeline.items().iter().any(|item| {
                    matches!(
                        item,
                        crate::tui::TimelineItem::User(message)
                            if !message.queued
                                && message.submission_id.as_deref()
                                    == Some(user_message.submission_id.as_str())
                    )
                }) {
                    suppress_session_event = true;
                }
            }
            SessionTransportEvent::AssistantDelta(_)
            | SessionTransportEvent::ReasoningDelta(_)
            | SessionTransportEvent::ToolPending(_)
            | SessionTransportEvent::ToolCancelled(_)
            | SessionTransportEvent::ToolStarted(_)
            | SessionTransportEvent::ToolOutputDelta(_) => {
                self.queued_prompt_lifecycle.clear_dispatch_ready();
            }
            SessionTransportEvent::TokenUsage(token_usage) => {
                let mut token_usage = token_usage.clone();
                self.state.merge_parent_prompt_composition(&mut token_usage);
                let request_output_tokens = token_usage.output_tokens;
                if request_output_tokens > 0 {
                    self.current_turn_output_tokens = self
                        .current_turn_output_tokens
                        .saturating_add(request_output_tokens);
                }
                token_usage.output_tokens = self.current_turn_output_tokens;
                self.state
                    .apply_event(SessionEvent::TokenUsage(token_usage));
                self.state.commit_sidebar_token_usage();
                suppress_session_event = true;
            }
            SessionTransportEvent::PreparedTokenUsage(token_usage) => {
                let mut token_usage = token_usage.clone();
                self.state.merge_parent_prompt_composition(&mut token_usage);
                self.state.apply_live_token_usage(token_usage.into());
                suppress_session_event = true;
            }
            SessionTransportEvent::CompactionStarted
                if self.state.toast().is_some_and(|toast| {
                    toast.message == self.state.t("runtime.context_organized")
                }) =>
            {
                self.state.toast = None;
            }
            SessionTransportEvent::CompactionCommitted { summary } => {
                let compacting_message = self.state.t("runtime.compacting_context");
                if self
                    .state
                    .toast()
                    .is_some_and(|toast| toast.message == compacting_message)
                {
                    self.state.toast = None;
                }
                if summary.is_none() && self.state.toast().is_none() {
                    self.show_toast(
                        self.state.t("runtime.context_organized"),
                        ToastKind::Success,
                    );
                }
            }
            SessionTransportEvent::SessionTokenUsage(token_usage) => {
                // A committed manual compaction replaces the request snapshot.
                // Its local estimate has no provider response/cache accounting.
                self.current_turn_output_tokens = 0;
                self.state
                    .apply_event(SessionEvent::TokenUsage(token_usage.clone()));
                self.state.commit_sidebar_token_usage();
                suppress_session_event = true;
            }
            SessionTransportEvent::ToolBatchFinished
                if !self.queued_prompts.is_empty()
                    && !self.queued_prompt_lifecycle.has_inflight_handoff() =>
            {
                self.queued_prompt_lifecycle.mark_dispatch_ready();
            }
            SessionTransportEvent::SessionResumed {
                session_id,
                branch_id,
                messages,
                records,
                evidence_count: _,
                model_id,
                token_usage,
                runtime_context,
                expert_models,
            } => {
                let _message_count = messages.len();
                if let Err(error) = self
                    .state
                    .try_replace_session_timeline_from_records_with_runtime_context(
                        records,
                        runtime_context.clone(),
                    )
                {
                    self.session_resume_pending = false;
                    self.state.show_toast(
                        self.state.t_fmt(
                            "runtime.context_projection_failed",
                            &[("error", &error.to_string())],
                        ),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state.clear_child_timeline_cache();
                self.session_resume_pending = false;
                self.output_rate_samples.clear();
                self.state.clear_all_output_token_rates();
                self.state.clear_pending_composer_settings();
                self.state.session_id = Some(session_id.clone());
                self.session_title = session_title_from_records(records);
                self.permission_lifecycle.clear_if_parent();
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.session_turn_active = false;
                self.current_turn_output_tokens = 0;
                self.state.timeline.remove_queued_user_message_previews();
                if let Some(model_id) = model_id {
                    self.apply_restored_model(model_id.clone());
                    self.state.set_provider_label_from_model_route(model_id);
                }
                for expert in &mut self.available_experts {
                    expert.route_id = expert_models
                        .get(&expert.agent_name)
                        .cloned()
                        .unwrap_or_else(|| self.state.model_id.clone());
                }
                if let Some(mode) = crate::transcript::restore_latest_permission_mode(records) {
                    self.state.set_permission_mode_label(mode);
                }
                self.state.set_current_context_branch(branch_id.clone());
                let resuming_message = self.state.t("runtime.resuming_session");
                if self
                    .state
                    .toast()
                    .is_some_and(|toast| toast.message == resuming_message)
                {
                    self.state.toast = None;
                }
                if let Some(token_usage) = token_usage {
                    self.state.set_token_usage(token_usage.clone().into());
                }
            }
            SessionTransportEvent::ParentSessionViewed {
                session_id,
                branch_id,
                records,
                model_id,
                token_usage,
                runtime_context,
            } => {
                // Pure view navigation, symmetrical to ChildSessionViewed: the parent
                // timeline is reprojected from transcript records, but the session is
                // still live, so queued submissions and in-flight runtime state are
                // preserved (unlike SessionResumed, which resets them). The timeline
                // projection clears the question dialog, so keep it aside first and
                // restore it when the in-flight handle is still live.
                let pending_question = self.state.pending_question.take();
                let parent_token_usage = self.state.model_token_usage.clone();
                let parent_output_token_rate = self.state.output_token_rate;
                let preserve_live_parent = self.session_turn_active
                    || self.state.phase == super::state::AppPhase::Running
                    || self.state.active_tool_call_id.is_some();
                let result = if preserve_live_parent {
                    self.state
                        .try_restore_parent_timeline_view_with_runtime_context(
                            records,
                            runtime_context.clone(),
                        )
                } else {
                    self.state
                        .try_replace_session_timeline_from_records_with_runtime_context(
                            records,
                            runtime_context.clone(),
                        )
                };
                if let Err(error) = result {
                    self.state.pending_question = pending_question;
                    self.state.show_toast(
                        self.state.t_fmt(
                            "runtime.context_projection_failed",
                            &[("error", &error.to_string())],
                        ),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state.session_id = Some(session_id.clone());
                self.session_title = session_title_from_records(records);
                // Transcript records do not contain queued submissions; republish
                // their previews so they remain visible and dispatchable.
                for prompt in &self.queued_prompts {
                    self.state.push_queued_user_message_preview(prompt.clone());
                }
                if self.pending_question_handle.is_some()
                    && let Some(question) = pending_question
                {
                    self.state.pending_question = Some(question);
                    self.state.phase = super::state::AppPhase::WaitingForPermission;
                }
                if let Some(model_id) = model_id {
                    let reasoning_effort_label = self.state.reasoning_effort_label.clone();
                    self.apply_restored_model(model_id.clone());
                    self.state
                        .set_reasoning_effort_label(reasoning_effort_label);
                    self.state.set_provider_label_from_model_route(model_id);
                }
                self.state.set_current_context_branch(branch_id.clone());
                if let Some(token_usage) = token_usage {
                    self.state.set_token_usage(token_usage.clone().into());
                } else if let Some(parent_token_usage) = parent_token_usage {
                    self.state.set_token_usage(parent_token_usage);
                }
                self.state.set_output_token_rate(parent_output_token_rate);
            }
            SessionTransportEvent::ContextBranchChanged { branch_id } => {
                self.state.set_current_context_branch(branch_id.clone());
            }
            SessionTransportEvent::ChildSessionViewed {
                parent_session_id,
                child_session_id,
                agent_name,
                index,
                total,
                pool_ordinal,
                records,
                runtime_context,
                in_progress_assistant_text,
            } => {
                if let Err(error) = self
                    .state
                    .try_replace_child_timeline_from_records_with_runtime_context(
                        records,
                        parent_session_id.clone(),
                        child_session_id.clone(),
                        agent_name.clone(),
                        *index,
                        *total,
                        *pool_ordinal,
                        runtime_context.clone(),
                        in_progress_assistant_text.clone(),
                    )
                {
                    self.state.show_toast(
                        self.state.t_fmt(
                            "runtime.context_projection_failed",
                            &[("error", &error.to_string())],
                        ),
                        ToastKind::Error,
                    );
                    return;
                }
            }
            SessionTransportEvent::SessionHistoryLoaded { entries } => {
                self.open_history_tree_dialog(entries);
            }
            SessionTransportEvent::SessionStarted {
                session_id,
                records,
                runtime_context,
                expert_models,
            } => {
                self.output_rate_samples.clear();
                self.state.clear_all_output_token_rates();
                if let Err(error) = self
                    .state
                    .try_replace_session_timeline_from_records_with_runtime_context(
                        records,
                        runtime_context.clone(),
                    )
                {
                    self.state.show_toast(
                        self.state.t_fmt(
                            "runtime.context_projection_failed",
                            &[("error", &error.to_string())],
                        ),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state.clear_child_timeline_cache();
                self.state.clear_pending_composer_settings();
                self.state.session_id = Some(session_id.clone());
                self.session_title = session_title_from_records(records);
                self.permission_lifecycle.clear();
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.session_turn_active = false;
                self.current_turn_output_tokens = 0;
                self.state.timeline.remove_queued_user_message_previews();
                for expert in &mut self.available_experts {
                    expert.route_id = expert_models
                        .get(&expert.agent_name)
                        .cloned()
                        .unwrap_or_else(|| self.state.model_id.clone());
                }
                // A newly created, still-empty session remains on the dashboard.
                self.state.active_session = false;
                self.state
                    .set_current_context_branch(crate::transcript::ROOT_CONTEXT_BRANCH_ID);
            }
            SessionTransportEvent::McpToolsDiscovered(servers) => {
                self.state.set_mcp_servers(servers.clone());
                self.refresh_open_mcp_dialog();
            }
            SessionTransportEvent::McpServerUpdated(server) => {
                self.state.update_mcp_server(server.clone());
                self.state
                    .set_mcp_server_updating(server.name.clone(), false);
                self.refresh_open_mcp_dialog();
                self.refresh_open_mcp_tools_dialog(&server.name);
            }
            SessionTransportEvent::McpServerUpdating { name, updating } => {
                self.state.set_mcp_server_updating(name.clone(), *updating);
                self.refresh_open_mcp_dialog();
            }
            SessionTransportEvent::McpServerToolsUpdated { name, tools } => {
                self.state.set_mcp_server_tools(name.clone(), tools.clone());
                self.refresh_open_mcp_tools_dialog(name);
            }
            SessionTransportEvent::McpDiscoveryUnavailable(error) => {
                self.state.mark_mcp_discovery_unavailable(error.clone());
                self.refresh_open_mcp_dialog();
            }
            SessionTransportEvent::McpDiagnostic(message) => {
                self.show_toast(message.clone(), ToastKind::Error);
            }
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name,
                parent_tool_call_id,
                event,
            } => {
                let mut event = event.clone();
                if matches!(
                    event,
                    SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
                ) {
                    self.output_rate_samples.retain(|sample| {
                        sample.child_session_id.as_deref() != Some(child_session_id)
                    });
                }
                self.state
                    .merge_child_prompt_composition(child_session_id, &mut event);
                if self.child_event_clears_pending_permission(child_session_id, &event) {
                    self.permission_lifecycle.clear();
                }
                if self.pending_question_child_session_id.as_deref() == Some(child_session_id)
                    && matches!(
                        event,
                        SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
                    )
                {
                    let _ = self.cancel_pending_question(
                        "question cancelled because the child session stopped",
                    );
                }
                self.state.apply_child_session_event_with_agent(
                    child_session_id,
                    agent_name.as_deref(),
                    parent_tool_call_id.as_deref(),
                    event,
                );
            }
            _ => {}
        }

        self.reproject_pending_permission();

        if !suppress_session_event && let Some(session_event) = event.session_event() {
            self.state.apply_event(session_event);
            self.reproject_pending_permission();
        }
    }
}

// ── 输入、命令分派与主题 ────────────────────────
impl TuiRuntime {
    fn reproject_pending_permission(&mut self) {
        self.state
            .set_pending_permission_projection(self.permission_lifecycle.projection());
    }

    fn approve_pending_permission(&mut self) -> Result<()> {
        if let Some(handle) = self.permission_lifecycle.take_handle() {
            handle.approve()?;
        }
        Ok(())
    }

    fn allow_always_pending_permission(&mut self) -> Result<()> {
        if self
            .state
            .pending_permission
            .as_ref()
            .is_some_and(|permission| permission.can_allow_always)
            && let Some(handle) = self.permission_lifecycle.take_handle()
        {
            handle.allow_always()?;
        }
        Ok(())
    }

    fn deny_pending_permission(&mut self) -> Result<()> {
        if let Some(handle) = self.permission_lifecycle.take_handle() {
            handle.deny()?;
        }
        Ok(())
    }

    fn pending_permission_matches_call(
        &self,
        call_id: &str,
        child_session_id: Option<&str>,
    ) -> bool {
        self.permission_lifecycle
            .matches_call(call_id, child_session_id)
    }

    fn child_event_clears_pending_permission(
        &self,
        child_session_id: &str,
        event: &SessionEvent,
    ) -> bool {
        self.permission_lifecycle
            .clears_for_child_event(child_session_id, event)
    }

    pub fn handle_input_action(&mut self, action: InputAction) -> Result<Option<RuntimeCommand>> {
        if !matches!(
            action,
            InputAction::Interrupt
                | InputAction::Quit
                | InputAction::Tick
                | InputAction::ChildPrefix
                | InputAction::ToggleSidebar
        ) {
            // Ctrl+C shares the interrupt confirmation while work is in flight, so the
            // pending confirmation must survive the second press.
            self.interrupt_confirmation_pending = false;
        }

        if !matches!(
            action,
            InputAction::NoOp
                | InputAction::Tick
                | InputAction::ChildPrefix
                | InputAction::ToggleSidebar
        ) {
            self.state.child_navigation_prefix = false;
        }

        if apply_edit_action(&mut self.state, &action) {
            if matches!(
                action,
                InputAction::Insert(_)
                    | InputAction::Paste(_)
                    | InputAction::PasteLongText(_)
                    | InputAction::InsertNewline
                    | InputAction::Backspace
                    | InputAction::Delete
            ) {
                self.reset_history_navigation();
            }
            return Ok(None);
        }

        match action {
            InputAction::SlashPanelNext => {
                self.select_next_slash_command();
                Ok(None)
            }
            InputAction::SlashPanelPrev => {
                self.select_previous_slash_command();
                Ok(None)
            }
            InputAction::SlashPanelAccept => {
                self.accept_selected_slash_command();
                Ok(None)
            }
            InputAction::SlashPanelDismiss => {
                self.state.dismiss_slash_panel();
                Ok(None)
            }
            InputAction::ScrollUp => {
                self.state.scroll_transcript_up(1);
                Ok(None)
            }
            InputAction::ScrollDown => {
                self.state.scroll_transcript_down(1);
                Ok(None)
            }
            InputAction::ScrollPageUp => {
                self.state.scroll_transcript_up(PAGE_SCROLL_ROWS);
                Ok(None)
            }
            InputAction::ScrollPageDown => {
                self.state.scroll_transcript_down(PAGE_SCROLL_ROWS);
                Ok(None)
            }
            InputAction::ScrollToBottom => {
                self.state.scroll_transcript_to_bottom();
                Ok(None)
            }
            InputAction::MouseScrollUp => {
                self.state.scroll_transcript_up(1);
                Ok(None)
            }
            InputAction::MouseScrollDown => {
                self.state.scroll_transcript_down(1);
                Ok(None)
            }
            InputAction::SidebarScrollUp => {
                self.state.scroll_sidebar_up(1);
                Ok(None)
            }
            InputAction::SidebarScrollDown => {
                self.state.scroll_sidebar_down(1);
                Ok(None)
            }
            InputAction::ToggleSidebarContext => {
                self.state.toggle_sidebar_context();
                Ok(None)
            }
            InputAction::ToggleSidebarMcp => {
                self.state.toggle_sidebar_mcp();
                Ok(None)
            }
            InputAction::CopySessionId => {
                self.handle_copy_session_id();
                Ok(None)
            }
            InputAction::ToggleSidebarTodos => {
                self.state.toggle_sidebar_todos();
                Ok(None)
            }
            InputAction::ToggleSidebar => {
                if self.state.is_read_only_child_view() {
                    self.state.child_navigation_prefix = false;
                    return Ok(None);
                }
                self.state.toggle_sidebar();
                self.state.child_navigation_prefix = false;
                self.persist_sidebar_preference();
                Ok(None)
            }
            InputAction::ShowModel => self.handle_shortcut_command(CommandIntent::ModelShow),
            InputAction::ShowPermission => {
                self.handle_shortcut_command(CommandIntent::PermissionShow)
            }
            InputAction::ShowReasoning => {
                self.handle_shortcut_command(CommandIntent::ReasoningShow)
            }
            InputAction::ShowThoughts => self.handle_shortcut_command(CommandIntent::ThoughtsShow),
            InputAction::ShowAgents => self.handle_shortcut_command(CommandIntent::AgentsShow),
            InputAction::ShowContext => self.handle_shortcut_command(CommandIntent::ContextBrowse),
            InputAction::ShowMcp => self.handle_shortcut_command(CommandIntent::McpBrowse),
            InputAction::ShowSkills => self.handle_shortcut_command(CommandIntent::SkillBrowse),
            InputAction::ShowHelp => self.handle_shortcut_command(CommandIntent::Help),
            InputAction::CycleReasoningEffort => {
                if self.state.is_read_only_child_view() {
                    Ok(None)
                } else {
                    Ok(self.cycle_reasoning_effort_command())
                }
            }
            InputAction::ChildPrefix => {
                self.state.child_navigation_prefix = true;
                self.state.child_navigation_prefix_ticks_remaining =
                    CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS;
                Ok(None)
            }
            InputAction::ChildFirst => Ok(Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::First,
                anchor_child_session_id: None,
            })),
            InputAction::ChildNext => Ok(Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::Next,
                anchor_child_session_id: None,
            })),
            InputAction::ChildPrev => Ok(Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::Prev,
                anchor_child_session_id: None,
            })),
            InputAction::HistorianView(view) => {
                if self.state.is_historian_child_view() {
                    let mut options = self.state.historian_report_options;
                    options.view = view;
                    self.state.set_historian_report_options(options);
                }
                Ok(None)
            }
            InputAction::HistorianSources => {
                if self.state.is_historian_child_view() {
                    let mut options = self.state.historian_report_options;
                    options.sources = !options.sources;
                    self.state.set_historian_report_options(options);
                }
                Ok(None)
            }
            InputAction::ChildParent => {
                if self.state.is_read_only_child_view() {
                    self.state.restore_parent_timeline_view();
                }
                Ok(Some(RuntimeCommand::ViewParent))
            }
            InputAction::QuestionPrevTab => {
                if let Some(question) = self.state.pending_question.as_mut() {
                    question.move_prev_tab();
                }
                Ok(None)
            }
            InputAction::QuestionNextTab => {
                if let Some(question) = self.state.pending_question.as_mut() {
                    question.move_next_tab();
                }
                Ok(None)
            }
            InputAction::QuestionPrevOption => {
                if let Some(question) = self.state.pending_question.as_mut() {
                    question.move_prev_row();
                }
                Ok(None)
            }
            InputAction::QuestionNextOption => {
                if let Some(question) = self.state.pending_question.as_mut() {
                    question.move_next_row();
                }
                Ok(None)
            }
            InputAction::QuestionPickOption(index) => {
                if let Some(question) = self.state.pending_question.as_mut() {
                    match question.pick_row(index.saturating_sub(1) as usize) {
                        QuestionAdvance::Submit => self.submit_pending_question()?,
                        QuestionAdvance::Editing
                        | QuestionAdvance::Advanced
                        | QuestionAdvance::None => {}
                    }
                }
                Ok(None)
            }
            InputAction::QuestionActivate => {
                enum Action {
                    BeginEdit,
                    Submit,
                    Advanced,
                    None,
                }

                let action = if let Some(question) = self.state.pending_question.as_mut() {
                    if question.editing_custom {
                        match question.commit_custom_answer() {
                            QuestionAdvance::Submit => Action::Submit,
                            QuestionAdvance::Advanced => Action::Advanced,
                            QuestionAdvance::Editing => Action::BeginEdit,
                            QuestionAdvance::None => Action::None,
                        }
                    } else {
                        match question.pick_row(question.active_row.index()) {
                            QuestionAdvance::Submit => Action::Submit,
                            QuestionAdvance::Advanced => Action::Advanced,
                            QuestionAdvance::Editing => Action::BeginEdit,
                            QuestionAdvance::None => Action::None,
                        }
                    }
                } else {
                    Action::None
                };

                match action {
                    Action::Submit => {
                        self.submit_pending_question()?;
                    }
                    Action::Advanced => {}
                    Action::BeginEdit | Action::None => {}
                }
                Ok(None)
            }
            InputAction::QuestionSubmit => {
                self.submit_pending_question()?;
                Ok(None)
            }
            InputAction::QuestionCancel => {
                let editing = self
                    .state
                    .pending_question
                    .as_ref()
                    .is_some_and(|question| question.editing_custom);
                if editing {
                    if let Some(question) = self.state.pending_question.as_mut() {
                        question.stop_custom_edit();
                    }
                } else {
                    self.state.toast = None;
                    self.cancel_pending_question("question dismissed by user")?;
                }
                Ok(None)
            }
            InputAction::QuestionInsert(ch) => {
                self.insert_pending_question_text(&ch.to_string());
                Ok(None)
            }
            InputAction::QuestionPaste(text) => {
                self.insert_pending_question_text(&text);
                Ok(None)
            }
            InputAction::QuestionBackspace => {
                self.backspace_pending_question_text();
                Ok(None)
            }
            InputAction::QuestionDelete => {
                self.delete_pending_question_text();
                Ok(None)
            }
            InputAction::QuestionMoveCursorLeft => {
                self.move_pending_question_cursor_left();
                Ok(None)
            }
            InputAction::QuestionMoveCursorRight => {
                self.move_pending_question_cursor_right();
                Ok(None)
            }
            InputAction::QuestionMoveCursorHome => {
                self.move_pending_question_cursor_home();
                Ok(None)
            }
            InputAction::QuestionMoveCursorEnd => {
                self.move_pending_question_cursor_end();
                Ok(None)
            }
            InputAction::DialogNext => {
                if let Some(dialog) = self.state.dialog_mut() {
                    if let Some(selected) = dialog.config_close_selected.as_mut() {
                        *selected = selected.saturating_add(1).min(2);
                    } else if dialog.kind == DialogKind::ConfigEditor
                        && dialog.config_expanded.is_some()
                    {
                        let last = dialog.config_detail_items.len().saturating_sub(1);
                        dialog.config_detail_selected =
                            dialog.config_detail_selected.saturating_add(1).min(last);
                    } else if dialog.kind == DialogKind::ContextPicker && dialog.detail_focused {
                        dialog.scroll_detail_next();
                    } else {
                        dialog.select_next();
                    }
                }
                self.sync_context_inspector_preview();
                self.preview_selected_theme();
                self.sync_config_dialog_description();
                Ok(None)
            }
            InputAction::DialogPrev => {
                if let Some(dialog) = self.state.dialog_mut() {
                    if let Some(selected) = dialog.config_close_selected.as_mut() {
                        *selected = selected.saturating_sub(1);
                    } else if dialog.kind == DialogKind::ConfigEditor
                        && dialog.config_expanded.is_some()
                    {
                        dialog.config_detail_selected =
                            dialog.config_detail_selected.saturating_sub(1);
                    } else if dialog.kind == DialogKind::ContextPicker && dialog.detail_focused {
                        dialog.scroll_detail_previous();
                    } else {
                        dialog.select_previous();
                    }
                }
                self.sync_context_inspector_preview();
                self.preview_selected_theme();
                self.sync_config_dialog_description();
                Ok(None)
            }
            InputAction::DialogToggleSessionScope => {
                self.toggle_session_scope();
                Ok(None)
            }
            InputAction::DialogAccept => self.handle_dialog_accept(),
            InputAction::DialogToggle => {
                if self
                    .state
                    .dialog()
                    .is_some_and(|dialog| matches!(dialog.kind, DialogKind::ExpertModelPicker(_)))
                {
                    if let Some(dialog) = self.state.dialog_mut()
                        && let Some(item) = dialog.items.get_mut(dialog.selected)
                    {
                        item.checked = !item.checked;
                    }
                    Ok(None)
                } else {
                    self.handle_mcp_toggle()
                }
            }
            InputAction::ConfigEditInsert(ch) => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    edit.buffer.insert(cursor, ch);
                    edit.cursor = cursor + ch.len_utf8();
                }
                Ok(None)
            }
            InputAction::ConfigEditPaste(text) => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    edit.buffer.insert_str(cursor, &text);
                    edit.cursor = cursor + text.len();
                }
                Ok(None)
            }
            InputAction::ConfigEditBackspace => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    if let Some((index, _)) = edit.buffer[..cursor].char_indices().next_back() {
                        edit.buffer.drain(index..cursor);
                        edit.cursor = index;
                    }
                }
                Ok(None)
            }
            InputAction::ConfigEditDelete => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    if let Some((_, ch)) = edit.buffer[cursor..].char_indices().next() {
                        edit.buffer.drain(cursor..cursor + ch.len_utf8());
                    }
                }
                Ok(None)
            }
            InputAction::ConfigEditLeft => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    edit.cursor = edit.buffer[..cursor]
                        .char_indices()
                        .next_back()
                        .map_or(0, |(index, _)| index);
                }
                Ok(None)
            }
            InputAction::ConfigEditRight => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    let cursor = edit.cursor.min(edit.buffer.len());
                    edit.cursor = edit.buffer[cursor..]
                        .char_indices()
                        .nth(1)
                        .map_or(edit.buffer.len(), |(index, _)| cursor + index);
                }
                Ok(None)
            }
            InputAction::ConfigEditHome => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    edit.cursor = 0;
                }
                Ok(None)
            }
            InputAction::ConfigEditEnd => {
                if let Some(edit) = self.state.config_edit.as_mut() {
                    edit.cursor = edit.buffer.len();
                }
                Ok(None)
            }
            InputAction::ConfigEditConfirm => {
                self.commit_config_edit();
                Ok(None)
            }
            InputAction::ConfigEditCancel => {
                self.state.config_edit = None;
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.config_error = None;
                }
                if let Some(index) = self.expanded_index() {
                    let selected = self
                        .state
                        .dialog()
                        .map(|dialog| dialog.config_detail_selected)
                        .unwrap_or(0);
                    self.reopen_config_detail(index, selected);
                }
                Ok(None)
            }
            InputAction::ConfigCollapse => {
                self.collapse_config_field();
                Ok(None)
            }
            InputAction::ConfigSave => {
                self.save_config_draft();
                Ok(None)
            }
            InputAction::ConfigListAppend => {
                self.append_config_list_item();
                Ok(None)
            }
            InputAction::ConfigListRemove => {
                match self.selected_config_field() {
                    Some(ConfigFieldRef::Table(path)) => self.remove_config_table(path),
                    _ => self.remove_config_list_item(),
                }
                Ok(None)
            }
            InputAction::DialogCancel => {
                if let Some(dialog) = self.state.dialog_mut()
                    && dialog.config_close_selected.take().is_some()
                {
                    return Ok(None);
                }
                if self
                    .state
                    .dialog()
                    .is_some_and(|dialog| dialog.kind == DialogKind::ConfigEditor)
                    && self.leave_config_table()
                {
                    return Ok(None);
                }
                if self.state.dialog().is_some_and(|dialog| {
                    dialog.kind == DialogKind::ConfigEditor && dialog.config_dirty
                }) {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.config_close_selected = Some(0);
                    }
                    return Ok(None);
                }
                if self.cancel_theme_preview() {
                    return Ok(None);
                }
                if let Some((query, selected_server)) = self
                    .state
                    .dialog()
                    .filter(|dialog| dialog.kind == DialogKind::McpToolsPicker)
                    .map(|dialog| {
                        (
                            dialog.mcp_primary_query.clone().unwrap_or_default(),
                            dialog.mcp_primary_selected_server.clone(),
                        )
                    })
                {
                    self.show_mcp_dialog_with_state(query, selected_server);
                    return Ok(None);
                }
                if let Some((query, selected_agent)) = self
                    .state
                    .dialog()
                    .filter(|dialog| matches!(dialog.kind, DialogKind::ExpertModelPicker(_)))
                    .map(|dialog| {
                        (
                            dialog.expert_primary_query.clone().unwrap_or_default(),
                            dialog.expert_primary_selected_agent.clone(),
                        )
                    })
                {
                    self.show_agents_dialog_with_state(query, selected_agent);
                    return Ok(None);
                }
                let detail_focused = self.state.dialog().is_some_and(|dialog| {
                    dialog.kind == DialogKind::ContextPicker && dialog.detail_focused
                });
                if detail_focused {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.detail_focused = false;
                        dialog.detail_scroll = 0;
                    }
                } else {
                    self.state.close_dialog();
                }
                Ok(None)
            }
            InputAction::DialogInsert(ch) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.insert_query_char(ch);
                }
                self.refresh_config_search();
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::DialogPaste(text) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    for ch in text.chars() {
                        dialog.insert_query_char(ch);
                    }
                }
                self.refresh_config_search();
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::DialogBackspace => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.pop_query_char();
                }
                self.refresh_config_search();
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::Submit => self.handle_submit(),
            InputAction::RemoveLastQueuedPrompt => {
                self.remove_last_queued_prompt();
                Ok(None)
            }
            InputAction::HistoryPrev => {
                self.navigate_history_previous();
                Ok(None)
            }
            InputAction::HistoryNext => {
                self.navigate_history_next();
                Ok(None)
            }
            InputAction::PermissionPrevOption => {
                self.state.move_permission_choice_prev();
                Ok(None)
            }
            InputAction::PermissionNextOption => {
                self.state.move_permission_choice_next();
                Ok(None)
            }
            InputAction::PermissionActivate => {
                match self.state.highlighted_permission_choice() {
                    Some(PermissionChoice::AllowOnce) => self.approve_pending_permission()?,
                    Some(PermissionChoice::AllowAlways) => {
                        self.allow_always_pending_permission()?
                    }
                    Some(PermissionChoice::Reject) => self.deny_pending_permission()?,
                    None => {}
                }
                Ok(None)
            }
            InputAction::ApprovePermission => {
                self.approve_pending_permission()?;
                Ok(None)
            }
            InputAction::ApprovePermissionAlways => {
                self.allow_always_pending_permission()?;
                Ok(None)
            }
            InputAction::DenyPermission => {
                self.deny_pending_permission()?;
                Ok(None)
            }
            InputAction::Interrupt => self.handle_interrupt(),
            InputAction::MouseSelectionStart(col, row) => {
                self.handle_selection_start(col, row);
                Ok(None)
            }
            InputAction::MouseSelectionDrag(col, row) => {
                self.handle_selection_drag(col, row);
                Ok(None)
            }
            InputAction::MouseSelectionEnd(col, row, activate_link) => {
                self.handle_selection_end(col, row, activate_link);
                Ok(None)
            }
            InputAction::ScrollbarDragStart(col, row) => {
                self.handle_scrollbar_drag_start(col, row);
                Ok(None)
            }
            InputAction::ScrollbarDragMove(_, row) => {
                self.handle_scrollbar_drag_move(row);
                Ok(None)
            }
            InputAction::ScrollbarDragEnd => {
                self.state.transcript_scrollbar_drag = None;
                Ok(None)
            }
            InputAction::CopySelection => {
                self.handle_copy_selection()?;
                Ok(None)
            }
            InputAction::PasteFromClipboard => {
                self.handle_paste_from_clipboard()?;
                Ok(None)
            }
            InputAction::ClearSelection => {
                self.state.text_selection = None;
                self.state.selection_in_progress = false;
                Ok(None)
            }
            InputAction::Quit => {
                // A running turn keeps control: Ctrl+C stops work in flight (same
                // confirmation as Esc) instead of ending the session and cancelling
                // its subagents. A child-session phase alone cannot block quitting.
                if self.engine_turn_is_active() {
                    return self.handle_interrupt();
                }
                let _ = self.cancel_pending_question(
                    "question cancelled because the application is quitting",
                );
                self.permission_lifecycle.clear();
                self.reproject_pending_permission();
                self.state.apply_event(SessionEvent::Quit);
                Ok(None)
            }
            InputAction::Tick => {
                if self.state.child_navigation_prefix {
                    if self.state.child_navigation_prefix_ticks_remaining > 0 {
                        self.state.child_navigation_prefix_ticks_remaining -= 1;
                    }
                    if self.state.child_navigation_prefix_ticks_remaining == 0 {
                        self.state.child_navigation_prefix = false;
                    }
                }
                self.poll_session_list();
                self.state.apply_event(SessionEvent::Tick);
                self.spinner_frame = self.spinner_frame.wrapping_add(1);
                self.tick_selection_autoscroll();
                Ok(None)
            }
            InputAction::Insert(_)
            | InputAction::Paste(_)
            | InputAction::PasteLongText(_)
            | InputAction::InsertNewline
            | InputAction::Backspace
            | InputAction::Delete
            | InputAction::MoveCursorLeft
            | InputAction::MoveCursorRight
            | InputAction::MoveCursorHome
            | InputAction::MoveCursorEnd
            | InputAction::NoOp => Ok(None),
        }
    }

    pub fn draw<D: RuntimeDrawer>(&mut self, drawer: &mut D) -> io::Result<()> {
        let now = std::time::Instant::now();
        self.sample_output_rate_graph(now);
        self.state.begin_live_presentations(now);
        self.state.refresh_live_presentations(now);
        drawer.draw(&mut self.state)
    }

    pub(super) fn clear_mcp_server_updating(&mut self, server_name: &str) {
        self.state
            .set_mcp_server_updating(server_name.to_string(), false);
        self.refresh_open_mcp_dialog();
    }

    fn clear_failed_pending_setting(&mut self, command: &crate::session::SessionCommand) {
        match command {
            crate::session::SessionCommand::SetModel(model_id) => {
                self.state.clear_pending_model_if(model_id);
            }
            crate::session::SessionCommand::SetReasoningEffort(effort) => {
                let label = reasoning_effort_status_label(Some(effort.clone()));
                self.state.clear_pending_reasoning_effort_if(&label);
            }
            crate::session::SessionCommand::SetPermissionMode(mode) => {
                self.state
                    .clear_pending_permission_mode_if(&mode.to_string());
            }
            crate::session::SessionCommand::SetFakeClient(_) => {
                // The optimistic fake badge is authoritative until the next
                // successful selection; a failed toggle leaves the prior state.
            }
            crate::session::SessionCommand::SetExpertAllowedModels { .. }
            | crate::session::SessionCommand::ToggleFastMode
            | crate::session::SessionCommand::ToggleMcpServer(_)
            | crate::session::SessionCommand::SubmitPrompt(_)
            | crate::session::SessionCommand::DelegateSubagent { .. }
            | crate::session::SessionCommand::Compact
            | crate::session::SessionCommand::ShowHistoryTree
            | crate::session::SessionCommand::Undo
            | crate::session::SessionCommand::Redo
            | crate::session::SessionCommand::NavigateHistory { .. }
            | crate::session::SessionCommand::ViewChild { .. }
            | crate::session::SessionCommand::ViewParent
            | crate::session::SessionCommand::ResumeSession(_)
            | crate::session::SessionCommand::NewSession
            | crate::session::SessionCommand::Interrupt => {}
        }
    }

    pub(super) fn project_deferred_setting(&mut self, command: &crate::session::SessionCommand) {
        match command {
            crate::session::SessionCommand::SetModel(model_id) => {
                let model_label = self
                    .available_models
                    .iter()
                    .find(|model| model.id == *model_id)
                    .map(|model| model.label.clone())
                    .unwrap_or_else(|| model_id.clone());
                self.state.set_pending_model(model_id.clone(), model_label);
            }
            crate::session::SessionCommand::SetReasoningEffort(effort) => {
                self.state
                    .set_pending_reasoning_effort(reasoning_effort_status_label(Some(
                        effort.clone(),
                    )));
            }
            crate::session::SessionCommand::SetPermissionMode(mode) => {
                self.state.set_pending_permission_mode(mode.to_string());
            }
            crate::session::SessionCommand::SetFakeClient(_) => {}
            crate::session::SessionCommand::SetExpertAllowedModels { .. }
            | crate::session::SessionCommand::ToggleFastMode
            | crate::session::SessionCommand::ToggleMcpServer(_)
            | crate::session::SessionCommand::SubmitPrompt(_)
            | crate::session::SessionCommand::DelegateSubagent { .. }
            | crate::session::SessionCommand::Compact
            | crate::session::SessionCommand::ShowHistoryTree
            | crate::session::SessionCommand::Undo
            | crate::session::SessionCommand::Redo
            | crate::session::SessionCommand::NavigateHistory { .. }
            | crate::session::SessionCommand::ViewChild { .. }
            | crate::session::SessionCommand::ViewParent
            | crate::session::SessionCommand::ResumeSession(_)
            | crate::session::SessionCommand::NewSession
            | crate::session::SessionCommand::Interrupt => {}
        }
    }

    pub(super) fn clear_unaccepted_queued_prompt(&mut self) {
        self.queued_prompt_lifecycle.clear_unaccepted();
    }

    fn engine_turn_is_active(&self) -> bool {
        has_active_or_pending_session_turn(active_turn_state(
            &self.state,
            self.session_turn_active,
            self.queued_prompt_lifecycle.has_inflight_handoff(),
            self.permission_lifecycle.is_pending(),
        ))
    }

    fn has_active_or_pending_session_turn(&self) -> bool {
        self.state.has_running_child_session() || self.engine_turn_is_active()
    }

    fn history_navigation_is_unavailable(&self) -> bool {
        self.has_active_or_pending_session_turn()
            || self.state.pending_question.is_some()
            || self.pending_question_handle.is_some()
            || !self.queued_prompts.is_empty()
    }

    fn handle_submit(&mut self) -> Result<Option<RuntimeCommand>> {
        if self.state.pending_permission.is_some() || self.state.pending_question.is_some() {
            return Ok(None);
        }

        if self.state.slash_panel_is_open()
            && let Some(selected) = self.selected_slash_command()
        {
            let current = self.state.input_buffer.trim();
            if current != selected.command {
                if !self.state.composer_tokens.is_empty() {
                    self.state
                        .show_toast(self.state.t("runtime.remove_attachments"), ToastKind::Info);
                    return Ok(None);
                }
                self.state.set_input(selected.insert_text);
                return Ok(None);
            }
        }

        let mut content = self.state.composer_content();
        content.trim_outer_text();
        if content.is_empty() {
            return Ok(None);
        }

        let prompt = content.text.clone();
        let command_input = self
            .state
            .input_buffer
            .replace(crate::tui::state::COMPOSER_ATTACHMENT_MARKER, "");

        let parsed_command = parse_command(&command_input);
        if !self.state.composer_tokens.is_empty()
            && !matches!(&parsed_command, Ok(CommandIntent::Prompt(_)))
        {
            self.state
                .show_toast(self.state.t("runtime.remove_attachments"), ToastKind::Info);
            return Ok(None);
        }
        self.reset_history_navigation();
        let active_session_turn = self.has_active_or_pending_session_turn();
        let active_turn_disposition = parsed_command.as_ref().ok().and_then(|intent| {
            crate::session::SessionCommand::from_command_intent(intent.clone())
                .map(|command| command.active_turn_disposition())
        });
        let active_turn_local_rejected = matches!(
            &parsed_command,
            Ok(CommandIntent::Exit | CommandIntent::ResumeShow)
        );
        let active_turn_local_command_allowed = matches!(
            &parsed_command,
            Ok(CommandIntent::Help
                | CommandIntent::PermissionShow
                | CommandIntent::ModelShow
                | CommandIntent::AgentsShow
                | CommandIntent::ReasoningShow
                | CommandIntent::ThoughtsShow
                | CommandIntent::ThoughtsSet(_)
                | CommandIntent::ToolsShow
                | CommandIntent::ToolsSet(_)
                | CommandIntent::ContextBrowse
                | CommandIntent::McpBrowse
                | CommandIntent::SkillBrowse
                | CommandIntent::TranscriptScrollbarSet(_)
                | CommandIntent::PanelSet(_)
                | CommandIntent::Theme(_)
                | CommandIntent::Fake(_))
        );
        if active_session_turn {
            match active_turn_disposition {
                Some(crate::session::ActiveTurnCommandDisposition::QueuePrompt)
                    if !self.state.is_read_only_child_view() =>
                {
                    self.queue_prompt(UserMessageSubmission::new(next_submission_id(), content));
                    return Ok(None);
                }
                Some(crate::session::ActiveTurnCommandDisposition::Reject)
                | Some(crate::session::ActiveTurnCommandDisposition::Interrupt)
                | None
                    if active_turn_local_rejected || !active_turn_local_command_allowed =>
                {
                    self.state
                        .show_toast(self.state.t("runtime.turn_running"), ToastKind::Info);
                    return Ok(None);
                }
                Some(crate::session::ActiveTurnCommandDisposition::QueuePrompt)
                | Some(crate::session::ActiveTurnCommandDisposition::Immediate)
                | Some(crate::session::ActiveTurnCommandDisposition::Defer)
                | Some(crate::session::ActiveTurnCommandDisposition::Reject)
                | Some(crate::session::ActiveTurnCommandDisposition::Interrupt)
                | None => {}
            }
        }

        if self.state.is_read_only_child_view() && !child_view_allows_prompt(&command_input) {
            return Ok(None);
        }

        if let Some(command) = self.handle_parsed_command(parsed_command)? {
            self.state.clear_input();
            return Ok(match command {
                SubmittedCommand::LocalOnly => None,
                SubmittedCommand::Runtime(command) => Some(command),
            });
        }

        let submitted_draft = composer_draft_for_submission(&self.state);
        self.state.clear_input();
        self.state.clear_composer_tokens();
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        self.queued_prompt_lifecycle.clear_dispatch_ready();
        self.session_turn_active = true;
        self.state.toast = None;
        self.submitted_prompts.push(prompt.clone());
        self.submitted_prompt_drafts.push(submitted_draft);

        Ok(Some(RuntimeCommand::SubmitPrompt(
            UserMessageSubmission::new(next_submission_id(), content),
        )))
    }

    fn navigate_history_previous(&mut self) {
        if self.submitted_prompts.is_empty() {
            return;
        }

        let next_index = match self.history_selection {
            Some(0) => 0,
            Some(index) => index.saturating_sub(1),
            None => {
                self.history_draft = Some(ComposerDraft {
                    input_buffer: self.state.input_buffer.clone(),
                    input_cursor: self.state.input_cursor,
                    tokens: self.state.composer_tokens.clone(),
                });
                self.submitted_prompts.len().saturating_sub(1)
            }
        };

        self.history_selection = Some(next_index);
        self.restore_submitted_prompt_draft(next_index);
    }

    fn navigate_history_next(&mut self) {
        let Some(index) = self.history_selection else {
            return;
        };

        if index + 1 < self.submitted_prompts.len() {
            let next_index = index + 1;
            self.history_selection = Some(next_index);
            self.restore_submitted_prompt_draft(next_index);
            return;
        }

        let draft = self.history_draft.take().unwrap_or(ComposerDraft {
            input_buffer: String::new(),
            input_cursor: 0,
            tokens: Vec::new(),
        });
        self.history_selection = None;
        self.state.input_buffer = draft.input_buffer;
        self.state.input_cursor = draft.input_cursor.min(self.state.input_buffer.len());
        self.state.composer_tokens = draft.tokens;
        self.state.assert_composer_token_invariant();
        self.state.sync_input_phase();
        self.state.sync_slash_panel();
    }

    fn restore_submitted_prompt_draft(&mut self, index: usize) {
        let Some(draft) = self.submitted_prompt_drafts.get(index).cloned() else {
            self.state.set_input(self.submitted_prompts[index].clone());
            return;
        };
        self.state.input_buffer = draft.input_buffer;
        self.state.input_cursor = draft.input_cursor.min(self.state.input_buffer.len());
        self.state.composer_tokens = draft.tokens;
        self.state.assert_composer_token_invariant();
        self.state.sync_input_phase();
        self.state.sync_slash_panel();
    }

    fn reset_history_navigation(&mut self) {
        self.history_selection = None;
        self.history_draft = None;
    }

    fn queue_prompt(&mut self, prompt: UserMessageSubmission) {
        let submitted_draft = composer_draft_for_submission(&self.state);
        self.state.clear_input();
        self.state.clear_composer_tokens();
        self.state.mark_session_active();
        self.submitted_prompts.push(prompt.content.text.clone());
        self.submitted_prompt_drafts.push(submitted_draft);
        self.queued_prompts.push_back(prompt.clone());
        self.state.push_queued_user_message_preview(prompt);
        self.state.toast = None;
    }

    fn remove_last_queued_prompt(&mut self) {
        if self.state.is_read_only_child_view() {
            return;
        }
        let Some(prompt) = self.queued_prompts.back() else {
            return;
        };
        if self.queued_prompt_lifecycle.dispatched_submission_id() == Some(prompt.id.as_str()) {
            return;
        }

        let submission_id = prompt.id.clone();
        self.queued_prompts.pop_back();
        self.state
            .timeline
            .remove_first_queued_user_message_preview(&submission_id);
    }

    fn take_next_queued_prompt_command(&mut self) -> Option<RuntimeCommand> {
        if !self.queued_prompt_lifecycle.is_dispatch_ready()
            || self.queued_prompt_lifecycle.has_inflight_handoff()
            || self.permission_lifecycle.is_pending()
            || self.state.pending_permission.is_some()
            || matches!(
                self.state.phase,
                super::state::AppPhase::WaitingForPermission | super::state::AppPhase::Quitting
            )
        {
            return None;
        }

        let prompt = self.queued_prompts.front()?.clone();
        self.queued_prompt_lifecycle.dispatch(prompt.clone());
        self.session_turn_active = true;
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        Some(RuntimeCommand::SubmitPrompt(prompt))
    }

    fn handle_interrupt(&mut self) -> Result<Option<RuntimeCommand>> {
        if !self.has_active_or_pending_session_turn() {
            self.interrupt_confirmation_pending = false;
            return Ok(None);
        }

        if !self.interrupt_confirmation_pending {
            self.interrupt_confirmation_pending = true;
            self.state.show_toast(
                self.state.t("runtime.press_again_to_interrupt"),
                ToastKind::Info,
            );
            return Ok(None);
        }

        self.interrupt_confirmation_pending = false;
        // Stop the visible stream now; the engine's Interrupted re-seals it idempotently.
        self.end_pacing();
        self.state.seal_active_reasoning(std::time::Instant::now());
        self.state
            .show_toast(self.state.t("runtime.interrupting"), ToastKind::Info);
        Ok(Some(RuntimeCommand::Interrupt))
    }

    fn handle_shortcut_command(&mut self, intent: CommandIntent) -> Result<Option<RuntimeCommand>> {
        self.handle_parsed_command(Ok(intent))?;
        Ok(None)
    }

    fn handle_parsed_command(
        &mut self,
        parsed: Result<CommandIntent, crate::command::CommandParseError>,
    ) -> Result<Option<SubmittedCommand>> {
        let intent = match parsed {
            Ok(intent) => intent,
            Err(error) => {
                let translator = self.state.translator();
                self.push_command_notice(error.render(&translator));
                return Ok(Some(SubmittedCommand::LocalOnly));
            }
        };

        // Backend-owned intents share classification with the CLI via SessionCommand.
        if let Some(session_command) =
            crate::session::SessionCommand::from_command_intent(intent.clone())
        {
            return self.handle_backend_session_command(session_command);
        }

        match intent {
            CommandIntent::Language(value) => self.handle_language_command(value),
            CommandIntent::Prompt(_) => Ok(None),
            CommandIntent::Exit => {
                self.state.apply_event(SessionEvent::Quit);
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::Help => {
                self.push_command_notice(help_summary(&self.state.translator()));
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::ModelShow => self.show_model_dialog(),
            CommandIntent::AgentsShow => self.show_agents_dialog(),
            CommandIntent::ReasoningShow => self.show_reasoning_dialog(),
            CommandIntent::ThoughtsShow => self.show_thoughts_dialog(),
            CommandIntent::ThoughtsSet(mode) => self.handle_thoughts_command(mode),
            CommandIntent::ToolsShow => self.show_tools_dialog(),
            CommandIntent::ToolsSet(mode) => self.handle_tools_command(mode),
            CommandIntent::PermissionShow => self.show_permission_dialog(),
            CommandIntent::Theme(command) => Ok(Some(self.handle_theme_command(command))),
            CommandIntent::Fake(command) => Ok(Some(self.handle_fake_command(command))),
            CommandIntent::TranscriptScrollbarSet(mode) => {
                Ok(Some(self.handle_transcript_scrollbar_command(mode)))
            }
            CommandIntent::PanelSet(mode) => Ok(Some(self.handle_panel_command(mode))),
            CommandIntent::ResumeShow => self.show_resume_dialog(),
            CommandIntent::ContextBrowse => self.show_context_dialog(),
            CommandIntent::McpBrowse => self.show_mcp_dialog(),
            CommandIntent::SkillBrowse => self.show_skill_dialog(),
            CommandIntent::ConfigShow => self.show_config_dialog(),
            CommandIntent::Delegate { .. }
            | CommandIntent::PermissionSet(_)
            | CommandIntent::ModelSet(_)
            | CommandIntent::FastToggle
            | CommandIntent::ReasoningSet(_)
            | CommandIntent::Compact
            | CommandIntent::Tree
            | CommandIntent::Undo
            | CommandIntent::Redo
            | CommandIntent::Resume(_)
            | CommandIntent::NewSession
            | CommandIntent::Child(_) => unreachable!(
                "backend-owned CommandIntent must map through SessionCommand::from_command_intent"
            ),
        }
    }

    fn handle_backend_session_command(
        &mut self,
        command: crate::session::SessionCommand,
    ) -> Result<Option<SubmittedCommand>> {
        use crate::session::SessionCommand;

        match command {
            SessionCommand::SubmitPrompt(_) => Ok(None),
            SessionCommand::SetModel(model_id) => self.handle_model_selection(model_id),
            SessionCommand::SetExpertAllowedModels {
                agent_name,
                model_ids,
            } => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::SetExpertAllowedModels {
                    agent_name,
                    model_ids,
                },
            ))),
            SessionCommand::ToggleFastMode => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ToggleFastMode,
            ))),
            SessionCommand::SetReasoningEffort(effort) => {
                Ok(Some(self.set_reasoning_effort_command(effort)))
            }
            SessionCommand::SetPermissionMode(mode) => {
                Ok(Some(self.set_permission_mode_command(mode)))
            }
            SessionCommand::SetFakeClient(client) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::SetFakeClient(client),
            ))),
            SessionCommand::Compact => Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Compact))),
            SessionCommand::ShowHistoryTree => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ShowHistoryTree,
            ))),
            SessionCommand::Undo => Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Undo))),
            SessionCommand::Redo => Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Redo))),
            SessionCommand::NavigateHistory { target_entry_id } => Ok(Some(
                SubmittedCommand::Runtime(RuntimeCommand::NavigateHistory { target_entry_id }),
            )),
            SessionCommand::ResumeSession(session_id) => {
                self.session_resume_pending = true;
                self.state
                    .show_toast(self.state.t("runtime.resuming_session"), ToastKind::Info);
                Ok(Some(SubmittedCommand::Runtime(
                    RuntimeCommand::ResumeSession(session_id),
                )))
            }
            SessionCommand::NewSession => {
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::NewSession)))
            }
            SessionCommand::ViewChild {
                navigation,
                anchor_child_session_id,
            } => {
                if navigation == SharedChildNavigation::Toggle
                    && self.state.transcript_view.is_child()
                {
                    self.state.restore_parent_timeline_view();
                    Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewParent)))
                } else {
                    let navigation = if navigation == SharedChildNavigation::Toggle {
                        SharedChildNavigation::First
                    } else {
                        navigation
                    };
                    Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewChild {
                        navigation,
                        anchor_child_session_id,
                    })))
                }
            }
            SessionCommand::ViewParent => {
                if self.state.transcript_view.is_child() {
                    self.state.restore_parent_timeline_view();
                }
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewParent)))
            }
            SessionCommand::DelegateSubagent { agent_name, task } => Ok(Some(
                SubmittedCommand::Runtime(RuntimeCommand::DelegateSubagent { agent_name, task }),
            )),
            SessionCommand::ToggleMcpServer(server_name) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ToggleMcpServer(server_name),
            ))),
            SessionCommand::Interrupt => {
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Interrupt)))
            }
        }
    }

    fn handle_language_command(
        &mut self,
        value: Option<String>,
    ) -> Result<Option<SubmittedCommand>> {
        let language = match value.as_deref() {
            Some(value) => crate::tui::i18n::Language::parse(value)
                .ok_or_else(|| anyhow!("Unsupported language"))
                .map(Some)?,
            None => None,
        };
        if let Some(language) = language {
            self.state.set_language(Some(language));
            TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
                prefs.language = Some(language.id().to_string());
            })
            .map_err(|error| anyhow!("failed to save language preference: {error}"))?;
        } else {
            let items = [
                DialogItem::new("en", "English", None),
                DialogItem::new("zh-CN", "简体中文", None),
            ];
            let mut dialog = DialogState::new(
                DialogKind::LanguagePicker,
                self.state.t("language.select"),
                None,
                items.into_iter().collect(),
            );
            dialog.selected = match self.state.language() {
                crate::tui::i18n::Language::En => 0,
                crate::tui::i18n::Language::ZhCn => 1,
            };
            self.state.open_dialog(dialog);
        }
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn handle_tools_command(&mut self, mode: ToolsDisplayMode) -> Result<Option<SubmittedCommand>> {
        self.apply_tools_display(mode);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn apply_tools_display(&mut self, mode: ToolsDisplayMode) {
        self.state.set_tools_display(mode);
        if let Err(error) = TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
            prefs.tools_display = mode;
        }) {
            tracing::warn!(%error, "failed to save tools display preference");
            self.state.show_toast(
                self.state.t("runtime.preference_not_saved"),
                ToastKind::Info,
            );
        }
    }

    fn handle_panel_command(&mut self, mode: PanelMode) -> SubmittedCommand {
        match mode {
            PanelMode::Toggle => self.state.toggle_sidebar(),
            PanelMode::Visible => self.state.set_sidebar_preference(false, true),
            PanelMode::Hidden => self.state.set_sidebar_preference(true, false),
        }
        self.persist_sidebar_preference();
        SubmittedCommand::LocalOnly
    }

    fn persist_sidebar_preference(&mut self) {
        if let Err(error) = TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
            prefs.sidebar_hidden = self.state.sidebar_hidden;
            prefs.sidebar_forced_open = self.state.sidebar_forced_open;
        }) {
            tracing::warn!(%error, "failed to save TUI preferences");
            self.state.show_toast(
                self.state.t("runtime.preference_not_saved"),
                ToastKind::Info,
            );
        }
    }

    fn handle_transcript_scrollbar_command(
        &mut self,
        mode: TranscriptScrollbarMode,
    ) -> SubmittedCommand {
        let visible = match mode {
            TranscriptScrollbarMode::Toggle => !self.state.transcript_scrollbar_visible,
            TranscriptScrollbarMode::Visible => true,
            TranscriptScrollbarMode::Hidden => false,
        };
        self.state.set_transcript_scrollbar_visible(visible);
        if let Err(error) = TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
            prefs.transcript_scrollbar_visible = visible;
        }) {
            tracing::warn!(%error, "failed to save transcript scrollbar preference");
            self.state.show_toast(
                self.state.t("runtime.preference_not_saved"),
                ToastKind::Info,
            );
        }
        SubmittedCommand::LocalOnly
    }

    fn handle_thoughts_command(
        &mut self,
        mode: ThoughtsDisplayMode,
    ) -> Result<Option<SubmittedCommand>> {
        self.apply_thoughts_display(mode);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn apply_thoughts_display(&mut self, mode: ThoughtsDisplayMode) {
        self.state.set_thoughts_display(mode);
        if let Err(error) = TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
            prefs.thoughts_display = mode;
        }) {
            tracing::warn!(%error, "failed to save TUI preferences");
            self.state.show_toast(
                self.state.t("runtime.preference_not_saved"),
                ToastKind::Info,
            );
        }
    }

    fn handle_theme_command(&mut self, command: ThemeCommand) -> SubmittedCommand {
        match command {
            ThemeCommand::Show => self.show_theme_dialog(),
            ThemeCommand::Set(theme) => self.apply_theme_selection(&theme),
        }
        SubmittedCommand::LocalOnly
    }

    fn handle_fake_command(&mut self, command: FakeCommand) -> SubmittedCommand {
        match command {
            FakeCommand::Show => {
                self.show_fake_dialog();
                SubmittedCommand::LocalOnly
            }
            FakeCommand::Set(client) => self.apply_fake_selection(client),
        }
    }

    fn show_fake_dialog(&mut self) {
        let current = self
            .state
            .fake_client
            .map(|client| client.as_str())
            .unwrap_or("off");
        let items = vec![
            DialogItem::new(
                "off",
                self.state.t("runtime.fake_mode_off"),
                Some(self.state.t("runtime.fake_mode_off_desc")),
            ),
            DialogItem::new(
                "auto",
                self.state.t("runtime.fake_mode_auto"),
                Some(self.state.t("runtime.fake_mode_auto_desc")),
            ),
            DialogItem::new(
                "codex",
                "Codex",
                Some(self.state.t("runtime.fake_mode_codex_desc")),
            ),
            DialogItem::new(
                "anthropic",
                "Anthropic",
                Some(self.state.t("runtime.fake_mode_anthropic_desc")),
            ),
        ];
        let mut dialog = DialogState::new(
            DialogKind::FakePicker,
            self.state.t("runtime.select_fake"),
            Some(self.state.t("runtime.choose_fake")),
            items,
        );
        dialog.selected = dialog
            .items
            .iter()
            .position(|item| item.id == current)
            .unwrap_or_default();
        self.state.open_dialog(dialog);
    }

    fn apply_fake_selection(
        &mut self,
        client: Option<crate::fake::FakeClient>,
    ) -> SubmittedCommand {
        SubmittedCommand::Runtime(RuntimeCommand::SetFakeClient(client))
    }

    fn show_theme_dialog(&mut self) {
        self.theme_preview_original = Some((self.state.theme_id.clone(), self.state.custom_theme));
        ensure_bundled_themes(&self.preferences_dir);
        let mut items = vec![
            DialogItem::new(
                "dark",
                "Dark",
                Some(self.state.t("runtime.theme_dark_desc")),
            ),
            DialogItem::new(
                "plain",
                "Plain",
                Some(self.state.t("runtime.theme_plain_desc")),
            ),
            DialogItem::new(
                "glass",
                "Glass",
                Some(self.state.t("runtime.theme_glass_desc")),
            ),
            DialogItem::new(
                "wireframe",
                "Wireframe",
                Some(self.state.t("runtime.theme_wireframe_desc")),
            ),
            DialogItem::new(
                "rainbow",
                "Rainbow",
                Some(self.state.t("runtime.theme_rainbow_desc")),
            ),
        ];
        for custom in discover_custom_themes(&self.preferences_dir) {
            let description = custom_theme_description(self.state(), &custom);
            items.push(DialogItem::new(custom.id, custom.label, description));
        }
        let mut dialog = DialogState::new(
            DialogKind::ThemePicker,
            self.state.t("runtime.select_theme"),
            Some(self.state.t("runtime.choose_palette")),
            items,
        );
        dialog.selected = dialog
            .items
            .iter()
            .position(|item| item.id == self.state.theme_id)
            .unwrap_or_default();
        self.state.open_dialog(dialog);
    }

    fn apply_theme_selection(&mut self, theme_id: &str) {
        self.theme_preview_original = None;
        if !self.activate_theme(theme_id) {
            return;
        }
        if let Err(error) = TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
            prefs.theme = self.state.theme_id.clone();
        }) {
            tracing::warn!(%error, "failed to save TUI preferences");
            self.state
                .show_toast(self.state.t("runtime.theme_not_saved"), ToastKind::Info);
        }
    }

    fn activate_theme(&mut self, theme_id: &str) -> bool {
        let Some(id) = normalize_theme_id(theme_id) else {
            self.state.show_toast(
                self.state
                    .t_fmt("runtime.invalid_theme", &[("theme", theme_id)]),
                ToastKind::Error,
            );
            return false;
        };
        if let Some(builtin) = ThemeName::parse(&id) {
            self.state.set_theme_name(builtin);
            return true;
        }
        ensure_bundled_themes(&self.preferences_dir);
        match load_custom_theme(&self.preferences_dir, &id) {
            Ok(palette) => {
                self.state.set_active_theme(id, Some(palette));
                true
            }
            Err(error) => {
                tracing::warn!(%error, theme = %id, "failed to load custom theme");
                self.state.show_toast(
                    self.state
                        .t_fmt("runtime.load_theme_failed", &[("theme", &id)]),
                    ToastKind::Error,
                );
                false
            }
        }
    }

    fn preview_selected_theme(&mut self) {
        let theme_id = self.state.dialog().and_then(|dialog| {
            (dialog.kind == DialogKind::ThemePicker)
                .then(|| dialog.selected_item())
                .flatten()
                .map(|item| item.id.clone())
        });
        if let Some(theme_id) = theme_id {
            let _ = self.activate_theme(&theme_id);
        }
    }

    fn cancel_theme_preview(&mut self) -> bool {
        if !self
            .state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::ThemePicker)
        {
            return false;
        }
        if let Some((theme_id, custom_theme)) = self.theme_preview_original.take() {
            self.state.set_active_theme(theme_id, custom_theme);
        }
        self.state.close_dialog();
        true
    }
}

// ── 对话框分派、模型与选择 ─────────────────────
impl TuiRuntime {
    fn show_permission_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = vec![
            DialogItem::new(
                "safe",
                self.state.t("permission.mode_safe"),
                Some(self.state.t("permission.desc_safe")),
            ),
            DialogItem::new(
                "default",
                self.state.t("permission.mode_default"),
                Some(self.state.t("permission.desc_default")),
            ),
            DialogItem::new(
                "auto",
                self.state.t("permission.mode_auto"),
                Some(self.state.t("permission.desc_auto")),
            ),
            DialogItem::new(
                "yolo",
                self.state.t("permission.mode_yolo"),
                Some(self.state.t("permission.desc_yolo")),
            ),
        ];
        let mut dialog = DialogState::new(
            DialogKind::PermissionPicker,
            self.state.t("permission.title"),
            Some(self.state.t("permission.subtitle")),
            items,
        );
        dialog.selected = match self
            .state
            .pending_composer_settings
            .permission_mode
            .as_deref()
            .unwrap_or(&self.state.permission_mode_label)
        {
            "safe" => 0,
            "auto" => 2,
            "yolo" => 3,
            _ => 1,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_config_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.load_config_draft();
        let (items, fields) = self.config_dialog_items();
        let mut dialog = DialogState::new(
            DialogKind::ConfigEditor,
            self.state.t("config.title"),
            None,
            items,
        );
        dialog.config_fields = fields;
        dialog.selected = 0;
        self.state.open_dialog(dialog);
        self.sync_config_dialog_description();
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn config_dialog_items(&self) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let Some(document) = self.config_draft.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let query = self
            .state
            .dialog()
            .map(|dialog| dialog.query.trim().to_string())
            .unwrap_or_default();
        if !query.is_empty() {
            return self.config_search_items(document, &query);
        }
        let level = self
            .state
            .dialog()
            .map(|dialog| dialog.config_path.clone())
            .unwrap_or_default();
        self.config_level_items(document, &level)
    }

    /// Rows for what is there, plus what the schema allows to add.
    fn config_level_items(
        &self,
        document: &toml_edit::DocumentMut,
        level: &[String],
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let Some(table) = Self::config_table(document, level) else {
            return (Vec::new(), Vec::new());
        };
        let section = level.join(".");
        let mut items = Vec::new();
        let mut fields = Vec::new();
        let mut present: Vec<String> = Vec::new();
        for (key, item) in table.iter() {
            present.push(key.to_string());
            let path: Vec<String> = level
                .iter()
                .cloned()
                .chain(std::iter::once(key.to_string()))
                .collect();
            if !Self::config_entry_visible(document, &path) {
                continue;
            }
            let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
            match item {
                toml_edit::Item::Table(_) => {
                    let count = self.config_table_size(document, &path);
                    items.push(
                        DialogItem::new(path.join("/"), key.to_string(), Some(count))
                            .with_section(section.clone()),
                    );
                    fields.push(ConfigFieldRef::Table(path));
                }
                toml_edit::Item::Value(value) => {
                    let entry = crate::config::leaf_config_entry(&path, value);
                    let field = if entry.kind == crate::config::ConfigEntryKind::Array {
                        ConfigFieldRef::List(path.clone())
                    } else if Self::config_field_values(document, &path).is_some() {
                        ConfigFieldRef::Choice(path.clone())
                    } else {
                        ConfigFieldRef::Field(path.clone())
                    };
                    let affordance = Self::config_affordance(&entry, &field);
                    items.push(
                        DialogItem::new(path.join("/"), entry.label, Some(entry.display))
                            .with_description(Self::config_description(&self.state, &path_refs))
                            .with_section(section.clone())
                            .with_right_detail(affordance),
                    );
                    fields.push(field);
                }
                _ => {}
            }
        }
        let level_refs: Vec<&str> = level.iter().map(String::as_str).collect();
        for name in crate::config::schema_properties(&level_refs) {
            if present.contains(&name) {
                continue;
            }
            let path: Vec<String> = level
                .iter()
                .cloned()
                .chain(std::iter::once(name.clone()))
                .collect();
            if !Self::config_entry_visible(document, &path) {
                continue;
            }
            let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
            items.push(
                DialogItem::new(path.join("/"), name, None)
                    .with_description(Self::config_description(&self.state, &path_refs))
                    .with_section(section.clone())
                    .with_right_detail("＋"),
            );
            fields.push(ConfigFieldRef::NewField(path));
        }
        let level_refs: Vec<&str> = level.iter().map(String::as_str).collect();
        if level.is_empty() {
            for table in crate::config::entry_tables(&[]) {
                if document.get(&table).is_some() {
                    continue;
                }
                items.push(
                    DialogItem::new(
                        format!("new/{table}"),
                        self.state.t_fmt("config.new_entry", &[("table", &table)]),
                        None,
                    )
                    .with_section(section.clone()),
                );
                fields.push(ConfigFieldRef::NewEntry(table));
            }
        } else if crate::config::table_accepts_entries(&level_refs) {
            let name = level.last().cloned().unwrap_or_default();
            items.push(
                DialogItem::new(
                    format!("new/{}", level.join("/")),
                    self.state.t_fmt("config.new_entry", &[("table", &name)]),
                    None,
                )
                .with_section(section.clone()),
            );
            fields.push(ConfigFieldRef::NewEntry(level.join("/")));
        }
        (items, fields)
    }

    /// Every matching leaf, flattened across tables, so search stays fast.
    fn config_search_items(
        &self,
        document: &toml_edit::DocumentMut,
        query: &str,
    ) -> (Vec<DialogItem>, Vec<ConfigFieldRef>) {
        let needle = query.to_lowercase();
        let mut items = Vec::new();
        let mut fields = Vec::new();
        for entry in crate::config::config_entries_in(document) {
            if !Self::config_entry_visible(document, &entry.path)
                || !entry.path.join(".").to_lowercase().contains(&needle)
            {
                continue;
            }
            let path: Vec<&str> = entry.path.iter().map(String::as_str).collect();
            let field = if entry.kind == crate::config::ConfigEntryKind::Array {
                ConfigFieldRef::List(entry.path.clone())
            } else if Self::config_field_values(document, &entry.path).is_some() {
                ConfigFieldRef::Choice(entry.path.clone())
            } else {
                ConfigFieldRef::Field(entry.path.clone())
            };
            let affordance = Self::config_affordance(&entry, &field);
            items.push(
                DialogItem::new(entry.path.join("."), entry.label, Some(entry.display))
                    .with_description(crate::config::field_schema(&path).map(|(fallback, key)| {
                        self.state
                            .t_opt(&format!("config.schema.{key}"))
                            .unwrap_or(fallback)
                    }))
                    .with_section(entry.section)
                    .with_right_detail(affordance),
            );
            fields.push(field);
        }
        (items, fields)
    }

    fn refresh_config_search(&mut self) {
        if self
            .state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::ConfigEditor)
        {
            self.refresh_config_dialog();
            self.sync_config_dialog_description();
        }
    }

    fn config_table<'a>(
        document: &'a toml_edit::DocumentMut,
        level: &[String],
    ) -> Option<&'a dyn toml_edit::TableLike> {
        let mut table: &dyn toml_edit::TableLike = document.as_table();
        for segment in level {
            table = table
                .get(segment)
                .and_then(toml_edit::Item::as_table_like)?;
        }
        Some(table)
    }

    fn config_description(state: &TuiState, path: &[&str]) -> Option<String> {
        crate::config::field_schema(path).map(|(fallback, key)| {
            state
                .t_opt(&format!("config.schema.{key}"))
                .unwrap_or(fallback)
        })
    }

    fn config_affordance(entry: &crate::config::ConfigEntry, field: &ConfigFieldRef) -> String {
        if entry.kind == crate::config::ConfigEntryKind::Bool {
            return if entry.display == "true" {
                "[x]"
            } else {
                "[ ]"
            }
            .to_string();
        }
        match field {
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) => "▸".to_string(),
            _ => String::new(),
        }
    }

    fn config_table_size(&self, document: &toml_edit::DocumentMut, table: &[String]) -> String {
        let count = crate::config::config_entries_in(document)
            .into_iter()
            .filter(|entry| entry.path.starts_with(table))
            .count()
            .to_string();
        self.state.t_fmt("config.field_count", &[("count", &count)])
    }

    fn persist_fast_mode(&self, enabled: bool) {
        if let Err(error) = crate::tui::preferences::TuiPreferences::update_in_dir(
            &self.preferences_dir,
            |preferences| preferences.fast_mode = enabled,
        ) {
            tracing::warn!(%error, "failed to persist fast mode");
        }
    }

    fn load_config_draft(&mut self) {
        self.config_draft = self.config_path.as_deref().and_then(|config_path| {
            std::fs::read_to_string(config_path)
                .ok()
                .and_then(|text| text.parse::<toml_edit::DocumentMut>().ok())
        });
    }

    fn edit_config_document(
        &mut self,
        edit: impl FnOnce(&mut toml_edit::DocumentMut) -> Result<()>,
    ) -> Result<()> {
        let Some(mut document) = self.config_draft.take() else {
            anyhow::bail!("configuration draft is unavailable");
        };
        let result = edit(&mut document);
        self.config_draft = Some(document);
        if result.is_ok()
            && let Some(dialog) = self.state.dialog_mut()
        {
            dialog.config_dirty = true;
            dialog.config_error = None;
        }
        result
    }

    fn save_config_draft(&mut self) -> bool {
        let Some(config_path) = self.config_path.clone() else {
            self.state
                .show_toast(self.state.t("config.unavailable"), ToastKind::Error);
            return false;
        };
        let Some(document) = self.config_draft.as_ref() else {
            return false;
        };
        match crate::config::save_config_document(&config_path, document) {
            Ok(()) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.config_dirty = false;
                    dialog.config_error = None;
                }
                self.state
                    .show_toast(self.state.t("config.saved_value"), ToastKind::Info);
                true
            }
            Err(error) => {
                self.report_config_save_failure(error);
                false
            }
        }
    }

    /// Fields whose siblings make them inapplicable stay out of the panel.
    fn config_entry_visible(document: &toml_edit::DocumentMut, path: &[String]) -> bool {
        if path.len() == 1 && path[0] == "fast_mode" {
            return false;
        }
        if path.len() == 3 && path[0] == "mcp" {
            let kind = crate::config::config_value_in(document, &["mcp", path[1].as_str(), "type"])
                .map(|(_, value)| value);
            match path[2].as_str() {
                "command" | "environment" | "env" => return kind.as_deref() != Some("remote"),
                "url" | "headers" | "oauth" => return kind.as_deref() == Some("remote"),
                _ => {}
            }
        }
        if path.len() == 6
            && path[0] == "providers"
            && path[2] == "models"
            && path[4] == "generation"
        {
            let flag = crate::config::config_value_in(
                document,
                &[
                    "providers",
                    path[1].as_str(),
                    "models",
                    path[3].as_str(),
                    "capabilities",
                    "generation",
                    path[5].as_str(),
                ],
            )
            .map(|(_, value)| value == "true")
            .unwrap_or(false);
            return flag;
        }
        if path.len() == 5
            && path[0] == "providers"
            && path[2] == "models"
            && path[4] == "protocol_settings"
        {
            let protocol = crate::config::config_value_in(
                document,
                &[
                    "providers",
                    path[1].as_str(),
                    "models",
                    path[3].as_str(),
                    "protocol",
                ],
            )
            .or_else(|| {
                crate::config::config_value_in(
                    document,
                    &["providers", path[1].as_str(), "protocol"],
                )
            })
            .map(|(_, value)| value);
            return protocol.as_deref() == Some("anthropic");
        }
        true
    }

    fn config_field_values(
        document: &toml_edit::DocumentMut,
        path: &[String],
    ) -> Option<Vec<String>> {
        if path.len() == 1 && path[0] == "active_provider" {
            return Self::config_table_names(document, &["providers"]);
        }
        if path.len() == 3 && path[0] == "providers" && path[2] == "default_model" {
            return Self::config_table_names(document, &["providers", &path[1], "models"]);
        }
        if (path.len() == 3 && path[0] == "agents" && path[2] == "reasoning_effort")
            || (path.len() == 6
                && path[0] == "providers"
                && path[2] == "models"
                && path[4] == "generation"
                && path[5] == "reasoning_effort")
        {
            return Some(
                crate::config::REASONING_EFFORTS
                    .iter()
                    .map(|value| value.to_string())
                    .collect(),
            );
        }
        if path.len() == 3 && path[0] == "agents" && path[2] == "allowed_models" {
            return Some(
                Self::config_table_names(document, &["providers"])?
                    .into_iter()
                    .flat_map(|provider| {
                        Self::config_table_names(document, &["providers", &provider, "models"])
                            .unwrap_or_default()
                            .into_iter()
                            .map(move |model| format!("{provider}/{model}"))
                    })
                    .collect(),
            );
        }
        if path.len() == 3 && path[0] == "agents" {
            match path[2].as_str() {
                "provider" => return Self::config_table_names(document, &["providers"]),
                "model" => {
                    let agent_provider = crate::config::config_value_in(
                        document,
                        &["agents", path[1].as_str(), "provider"],
                    )
                    .or_else(|| crate::config::config_value_in(document, &["active_provider"]))
                    .map(|(_, value)| value)
                    .or_else(|| {
                        Self::config_table_names(document, &["providers"])?
                            .into_iter()
                            .next()
                    })?;
                    return Self::config_table_names(
                        document,
                        &["providers", &agent_provider, "models"],
                    );
                }
                _ => {}
            }
        }
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        crate::config::field_enum(&path_refs)
    }

    fn config_table_names(document: &toml_edit::DocumentMut, path: &[&str]) -> Option<Vec<String>> {
        let path: Vec<String> = path.iter().map(|segment| segment.to_string()).collect();
        Some(
            Self::config_table(document, &path)?
                .iter()
                .filter(|(_, item)| item.as_table_like().is_some())
                .map(|(name, _)| name.to_string())
                .collect(),
        )
    }

    fn expanded_field(&self) -> Option<ConfigFieldRef> {
        let dialog = self.state.dialog()?;
        dialog.config_fields.get(dialog.config_expanded?).cloned()
    }

    fn selected_config_field(&self) -> Option<ConfigFieldRef> {
        let dialog = self.state.dialog()?;
        dialog.config_fields.get(dialog.selected).cloned()
    }

    fn expanded_index(&self) -> Option<usize> {
        self.state
            .dialog()
            .and_then(|dialog| dialog.config_expanded)
    }

    fn expand_config_field(&mut self, index: usize) {
        let Some(field) = self
            .state
            .dialog()
            .and_then(|dialog| dialog.config_fields.get(index).cloned())
        else {
            return;
        };
        let (ConfigFieldRef::List(path) | ConfigFieldRef::Choice(path)) = &field else {
            return;
        };
        let Some(document) = self.config_draft.as_ref() else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let array = crate::config::config_array_in(document, &path_refs);
        let values = match &array {
            Some(values) => values.clone(),
            None => {
                let Some(values) = Self::config_field_values(document, path) else {
                    return;
                };
                values
            }
        };
        let current = self
            .state
            .dialog()
            .and_then(|dialog| dialog.selected_item())
            .and_then(|item| item.detail.clone());
        let mut items = values
            .iter()
            .map(|value| DialogItem::new(value.clone(), value.clone(), None))
            .collect::<Vec<_>>();
        if matches!(field, ConfigFieldRef::Choice(_))
            && Self::config_field_allows_custom(document, path)
        {
            items.push(DialogItem::new(
                CONFIG_CUSTOM_CHOICE.to_string(),
                self.state.t("config.custom_value"),
                None,
            ));
        }
        let selected = current
            .and_then(|current| items.iter().position(|item| item.label == current))
            .unwrap_or(0);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_expanded = Some(index);
            dialog.config_detail_items = items;
            dialog.config_detail_selected = selected;
            dialog.config_detail_target = None;
        }
    }

    fn collapse_config_field(&mut self) {
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_expanded = None;
            dialog.config_detail_items.clear();
            dialog.config_detail_selected = 0;
            dialog.config_detail_target = None;
        }
    }

    /// A field whose schema is an open string still accepts hand-written values.
    fn config_field_allows_custom(document: &toml_edit::DocumentMut, path: &[String]) -> bool {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        crate::config::field_enum(&path_refs).is_none()
            && Self::config_field_values(document, path).is_some()
    }

    fn config_list_has_choices(&self, path: &[String]) -> bool {
        self.config_draft
            .as_ref()
            .is_some_and(|document| Self::config_field_values(document, path).is_some())
    }

    fn show_config_element_choices(&mut self, path: &[String], index: usize) {
        let Some(document) = self.config_draft.as_ref() else {
            return;
        };
        let Some(values) = Self::config_field_values(document, path) else {
            return;
        };
        let items = values
            .iter()
            .map(|value| DialogItem::new(value.clone(), value.clone(), None))
            .collect::<Vec<_>>();
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_detail_items = items;
            dialog.config_detail_selected = 0;
            dialog.config_detail_target = Some(index);
        }
    }

    fn reopen_config_detail(&mut self, index: usize, selected: usize) {
        self.collapse_config_field();
        self.expand_config_field(index);
        if let Some(dialog) = self.state.dialog_mut() {
            let last = dialog.config_detail_items.len().saturating_sub(1);
            dialog.config_detail_selected = selected.min(last);
        }
    }

    fn accept_config_detail(&mut self) {
        let Some(field) = self.expanded_field() else {
            return;
        };
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        let selected = dialog.config_detail_selected;
        let target = dialog.config_detail_target;
        let Some(item) = dialog.config_detail_items.get(selected).cloned() else {
            return;
        };
        let value = item.label.clone();
        match field {
            ConfigFieldRef::List(path) => match target {
                Some(index) => self.commit_config_list_item(&path, index, &value),
                None if self.config_list_has_choices(&path) => {
                    self.show_config_element_choices(&path, selected)
                }
                None => self.begin_config_list_item_edit(ConfigFieldRef::ListItem(path, selected)),
            },
            ConfigFieldRef::Choice(path) => {
                if item.id == CONFIG_CUSTOM_CHOICE {
                    self.collapse_config_field();
                    self.begin_config_edit(&path);
                } else {
                    self.apply_config_choice(&path, &value);
                }
            }
            _ => {}
        }
    }

    fn apply_config_choice(&mut self, path: &[String], value: &str) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::String(value.to_string()),
            )
        });
        match result {
            Ok(()) => {
                self.collapse_config_field();
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn append_config_list_item(&mut self) {
        let Some(ConfigFieldRef::List(path)) = self.expanded_field() else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let next = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
            .map(|values| values.len())
            .unwrap_or(0);
        if self.config_list_has_choices(&path) {
            self.show_config_element_choices(&path, next);
            return;
        }
        let field = ConfigFieldRef::ListItem(path, next);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog
                .config_detail_items
                .push(DialogItem::new(String::new(), String::new(), None));
            dialog.config_detail_selected = dialog.config_detail_items.len() - 1;
        }
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field,
            cursor: 0,
            buffer: String::new(),
        });
    }

    fn remove_config_list_item(&mut self) {
        let Some(ConfigFieldRef::List(path)) = self.expanded_field() else {
            return;
        };
        let Some(index) = self
            .state
            .dialog()
            .map(|dialog| dialog.config_detail_selected)
        else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let Some(mut values) = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
        else {
            return;
        };
        if index >= values.len() {
            return;
        }
        values.remove(index);
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::Array(values),
            )
        });
        match result {
            Ok(()) => {
                if let Some(expanded) = self.expanded_index() {
                    self.reopen_config_detail(expanded, index);
                }
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn begin_config_list_item_edit(&mut self, field: ConfigFieldRef) {
        let ConfigFieldRef::ListItem(path, index) = &field else {
            return;
        };
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let buffer = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
            .and_then(|values| values.get(*index).cloned())
            .unwrap_or_default();
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field,
            cursor: buffer.len(),
            buffer,
        });
    }

    fn commit_config_list_item(&mut self, path: &[String], index: usize, value: &str) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let Some(mut values) = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_array_in(document, &path_refs))
        else {
            return;
        };
        let selected = if index >= values.len() {
            values.push(value.to_string());
            values.len() - 1
        } else {
            values[index] = value.to_string();
            index
        };
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path_refs,
                crate::config::ConfigScalar::Array(values),
            )
        });
        match result {
            Ok(()) => {
                if let Some(expanded) = self.expanded_index() {
                    self.reopen_config_detail(expanded, selected);
                }
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn commit_config_new_entry(&mut self, table: &str, name: &str) {
        if name.is_empty() {
            return;
        }
        let path: Vec<String> = table
            .split('/')
            .map(str::to_string)
            .chain(std::iter::once(name.to_string()))
            .collect();
        self.commit_config_new_item(&path);
    }

    fn commit_config_new_field(&mut self, path: &[String]) {
        self.commit_config_new_item(path);
    }

    fn commit_config_new_item(&mut self, path: &[String]) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let item = crate::config::default_value(&path_refs)
            .unwrap_or_else(|| toml_edit::Item::Table(toml_edit::Table::new()));
        let result = self.edit_config_document(|document| {
            crate::config::set_config_item(document, &path_refs, item)
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn sync_config_dialog_description(&mut self) {
        let description = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::ConfigEditor)
            .and_then(|dialog| dialog.selected_item())
            .and_then(|item| item.description.clone());
        if let Some(dialog) = self.state.dialog_mut()
            && dialog.kind == DialogKind::ConfigEditor
        {
            dialog.description = description;
        }
    }

    fn handle_config_editor_accept(&mut self) {
        if let Some(selected) = self
            .state
            .dialog()
            .and_then(|dialog| dialog.config_close_selected)
        {
            match selected {
                0 => {
                    if self.save_config_draft() {
                        self.state.close_dialog();
                        self.config_draft = None;
                    } else if let Some(dialog) = self.state.dialog_mut() {
                        dialog.config_close_selected = None;
                    }
                }
                1 => {
                    self.state.close_dialog();
                    self.config_draft = None;
                    self.state
                        .show_toast(self.state.t("config.discarded"), ToastKind::Info);
                }
                _ => {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.config_close_selected = None;
                    }
                }
            }
            return;
        }
        if self.expanded_index().is_some() {
            self.accept_config_detail();
            return;
        }
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        let index = dialog.selected;
        let detail = dialog.selected_item().and_then(|item| item.detail.clone());
        let Some(field) = dialog.config_fields.get(index).cloned() else {
            return;
        };
        match field {
            ConfigFieldRef::NewEntry(table) => {
                self.state.config_edit = Some(crate::tui::state::ConfigEditState {
                    field: ConfigFieldRef::NewEntry(table),
                    cursor: 0,
                    buffer: String::new(),
                });
            }
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) => self.expand_config_field(index),
            ConfigFieldRef::Field(path) => {
                let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
                let kind = self
                    .config_draft
                    .as_ref()
                    .and_then(|document| crate::config::config_value_in(document, &path_refs))
                    .map(|(kind, _)| kind);
                match kind {
                    Some(crate::config::ConfigEntryKind::Bool) => {
                        self.toggle_config_bool(&path, detail.as_deref() == Some("true"))
                    }
                    Some(crate::config::ConfigEntryKind::Text)
                    | Some(crate::config::ConfigEntryKind::Integer)
                    | Some(crate::config::ConfigEntryKind::Float) => self.begin_config_edit(&path),
                    _ => {}
                }
            }
            ConfigFieldRef::NewField(path) => self.commit_config_new_field(&path),
            ConfigFieldRef::Table(path) => self.enter_config_table(path),
            ConfigFieldRef::ListItem(..) => {}
        }
    }

    fn enter_config_table(&mut self, path: Vec<String>) {
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_selected.push(dialog.selected);
            dialog.config_path = path;
            dialog.selected = 0;
            dialog.query.clear();
        }
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
    }

    fn leave_config_table(&mut self) -> bool {
        let Some(dialog) = self.state.dialog_mut() else {
            return false;
        };
        if dialog.config_path.is_empty() {
            return false;
        }
        dialog.config_path.pop();
        dialog.selected = dialog.config_selected.pop().unwrap_or(0);
        self.refresh_config_dialog();
        self.sync_config_dialog_description();
        true
    }

    fn remove_config_table(&mut self, path: Vec<String>) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::remove_config_table(document, &path_refs)
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn toggle_config_bool(&mut self, segments: &[String], current: bool) {
        let path: Vec<&str> = segments.iter().map(String::as_str).collect();
        let result = self.edit_config_document(|document| {
            crate::config::set_config_scalar(
                document,
                &path,
                crate::config::ConfigScalar::Bool(!current),
            )
        });
        match result {
            Ok(()) => {
                self.refresh_config_dialog();
                self.state
                    .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
            }
            Err(error) => self.report_config_save_failure(error),
        }
    }

    fn begin_config_edit(&mut self, path: &[String]) {
        let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let buffer = self
            .config_draft
            .as_ref()
            .and_then(|document| crate::config::config_value_in(document, &path_refs))
            .map(|(_, text)| text)
            .unwrap_or_default();
        self.state.config_edit = Some(crate::tui::state::ConfigEditState {
            field: ConfigFieldRef::Field(path.to_vec()),
            cursor: buffer.len(),
            buffer,
        });
    }

    fn commit_config_edit(&mut self) {
        let Some(edit) = self.state.config_edit.clone() else {
            return;
        };
        let value = edit.buffer.trim().to_string();
        match edit.field {
            ConfigFieldRef::NewEntry(table) => {
                self.state.config_edit = None;
                self.commit_config_new_entry(&table, &value);
            }
            ConfigFieldRef::ListItem(path, index) => {
                self.state.config_edit = None;
                self.commit_config_list_item(&path, index, &value);
            }
            ConfigFieldRef::NewField(path) => {
                self.state.config_edit = None;
                self.commit_config_new_field(&path);
            }
            ConfigFieldRef::Field(path) => {
                let path_refs: Vec<&str> = path.iter().map(String::as_str).collect();
                let kind = self
                    .config_draft
                    .as_ref()
                    .and_then(|document| crate::config::config_value_in(document, &path_refs))
                    .map(|(kind, _)| kind)
                    .unwrap_or(crate::config::ConfigEntryKind::Text);
                let scalar = match kind {
                    crate::config::ConfigEntryKind::Integer => match value.parse::<i64>() {
                        Ok(number) => crate::config::ConfigScalar::Integer(number),
                        Err(_) => {
                            let message = self.state.t("config.invalid_number");
                            if let Some(dialog) = self.state.dialog_mut() {
                                dialog.config_error = Some(message.clone());
                            }
                            self.state.show_toast(message, ToastKind::Error);
                            return;
                        }
                    },
                    crate::config::ConfigEntryKind::Float => match value.parse::<f64>() {
                        Ok(number) => crate::config::ConfigScalar::Float(number),
                        Err(_) => {
                            let message = self.state.t("config.invalid_float");
                            if let Some(dialog) = self.state.dialog_mut() {
                                dialog.config_error = Some(message.clone());
                            }
                            self.state.show_toast(message, ToastKind::Error);
                            return;
                        }
                    },
                    _ => crate::config::ConfigScalar::String(value.clone()),
                };
                let result = self.edit_config_document(|document| {
                    crate::config::set_config_scalar(document, &path_refs, scalar)
                });
                match result {
                    Ok(()) => {
                        self.state.config_edit = None;
                        self.refresh_config_dialog();
                        self.state
                            .show_toast(self.state.t("config.unsaved"), ToastKind::Info);
                    }
                    Err(error) => self.report_config_save_failure(error),
                }
            }
            ConfigFieldRef::Choice(_) | ConfigFieldRef::List(_) | ConfigFieldRef::Table(_) => {}
        }
    }

    fn report_config_save_failure(&mut self, error: anyhow::Error) {
        tracing::warn!(%error, "failed to persist configuration change");
        let error_text = format!("{error:#}");
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.config_error = Some(error_text.clone());
        }
        self.state.show_toast(
            self.state
                .t_fmt("config.save_failed", &[("error", &error_text)]),
            ToastKind::Error,
        );
    }

    fn refresh_config_dialog(&mut self) {
        let (items, fields) = self.config_dialog_items();
        if let Some(dialog) = self.state.dialog_mut() {
            let selected = dialog.selected.min(items.len().saturating_sub(1));
            dialog.items = items;
            dialog.config_fields = fields;
            dialog.selected = selected;
        }
    }

    fn apply_model_catalog_update(&mut self, catalog: &ModelCatalogUpdatedEvent) {
        self.available_models = catalog
            .models
            .iter()
            .map(AvailableModel::from_catalog_entry)
            .collect();

        let items = self.model_dialog_items();
        let Some(dialog) = self.state.dialog_mut() else {
            return;
        };
        if !matches!(
            dialog.kind,
            DialogKind::ModelPicker | DialogKind::ExpertModelPicker(_)
        ) {
            return;
        }
        let selected_id = dialog.selected_item().map(|item| item.id.clone());
        let checked_ids = matches!(dialog.kind, DialogKind::ExpertModelPicker(_)).then(|| {
            dialog
                .items
                .iter()
                .filter(|item| item.checked)
                .map(|item| item.id.clone())
                .collect::<std::collections::HashSet<_>>()
        });
        let query = dialog.query.clone();
        let old_selected = dialog.selected;
        dialog.items = items;
        if let Some(checked_ids) = checked_ids {
            for item in &mut dialog.items {
                item.checked = checked_ids.contains(&item.id);
            }
        }
        dialog.query = query;
        dialog.selected = selected_id
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| old_selected.min(dialog.items.len().saturating_sub(1)));
    }

    fn model_dialog_items(&self) -> Vec<DialogItem> {
        self.available_models
            .iter()
            .map(|model| {
                DialogItem::new(model.id.clone(), model.label.clone(), None)
                    .with_section(model.provider.clone())
            })
            .collect()
    }

    fn show_model_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = self.model_dialog_items();
        let mut dialog = DialogState::new(
            DialogKind::ModelPicker,
            self.state.t("runtime.select_model"),
            None,
            items,
        );
        let selected_model_id = self
            .state
            .pending_composer_settings
            .model
            .as_ref()
            .map(|(model_id, _)| model_id.as_str())
            .unwrap_or(&self.state.model_id);
        if let Some(index) = self
            .available_models
            .iter()
            .position(|model| model.id == selected_model_id)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_agents_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.show_agents_dialog_with_state(String::new(), None);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    /// Rebuilds an open expert list so its route summaries track the experts' live routes.
    fn refresh_open_agent_picker(&mut self) {
        let Some((query, selected_agent)) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::AgentPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
        else {
            return;
        };
        self.show_agents_dialog_with_state(query, selected_agent);
    }

    fn show_agents_dialog_with_state(&mut self, query: String, selected_agent: Option<String>) {
        let items = self
            .available_experts
            .iter()
            .map(|expert| {
                DialogItem::new(expert.agent_name.clone(), expert.agent_name.clone(), None)
                    .with_section(self.state.t("ui.experts"))
                    .with_right_detail(expert.model_summary())
            })
            .collect();
        let mut dialog = DialogState::new(
            DialogKind::AgentPicker,
            self.state.t("ui.expert_models"),
            None,
            items,
        );
        dialog.query = query;
        if let Some(agent_name) = selected_agent
            && let Some(index) = dialog.items.iter().position(|item| item.id == agent_name)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    fn show_expert_model_dialog(&mut self, agent_name: String) {
        let (primary_query, primary_selected_agent) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::AgentPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
            .unwrap_or_default();
        let expert = self
            .available_experts
            .iter()
            .find(|expert| expert.agent_name == agent_name);
        let current_route = expert
            .map(|expert| expert.route_id.clone())
            .unwrap_or_else(|| self.state.model_id.clone());
        let allowed_models = expert
            .map(|expert| expert.allowed_models.clone())
            .unwrap_or_default();
        let mut dialog = DialogState::new(
            DialogKind::ExpertModelPicker(agent_name),
            self.state.t("runtime.select_expert_model"),
            None,
            self.model_dialog_items()
                .into_iter()
                .map(|item| {
                    let checked = allowed_models.contains(&item.id);
                    item.with_checked(checked)
                })
                .collect(),
        );
        dialog.expert_primary_query = Some(primary_query);
        dialog.expert_primary_selected_agent = primary_selected_agent;
        if let Some(index) = dialog
            .items
            .iter()
            .position(|item| item.id == current_route)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    fn apply_restored_model(&mut self, model_id: String) {
        if let Some(model) = self
            .available_models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
        {
            self.state.set_model(model.id.clone(), model.label.clone());
            self.state
                .set_model_context_window(model.context_window_tokens);
            self.state
                .set_reasoning_effort_label(Some(reasoning_effort_status_label(
                    model.reasoning_effort,
                )));
        } else {
            self.state.set_model(model_id.clone(), model_id);
            self.state.set_model_context_window(None);
            self.state.set_reasoning_effort_label(None);
        }
    }

    fn handle_model_selection(&mut self, model_id: String) -> Result<Option<SubmittedCommand>> {
        let Some(model) = self
            .available_models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
        else {
            let available = self
                .available_models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            self.push_command_notice(self.state.t_fmt(
                "runtime.unknown_model",
                &[("model", &model_id), ("available", &available)],
            ));
            return Ok(Some(SubmittedCommand::LocalOnly));
        };

        Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::SetModel(
            model.id,
        ))))
    }

    fn session_rows(
        &self,
        sessions: &[SessionSummary],
        scope: SessionPickerScope,
    ) -> Vec<DialogItem> {
        session_dialog_items(
            sessions,
            self.workspace_key.as_deref(),
            scope,
            &self.state.t("dialog.session_unassigned"),
        )
    }

    fn toggle_session_scope(&mut self) {
        let Some(scope) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::SessionPicker)
            .map(|dialog| dialog.session_scope.toggled())
        else {
            return;
        };
        let items = self.session_rows(&self.session_summaries, scope);
        if let Some(dialog) = self.state.dialog_mut() {
            dialog.session_scope = scope;
            dialog.replace_items(items);
        }
    }

    fn show_resume_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        if self.session_list_rx.is_some() {
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.session_list_rx = Some(rx);
        let sessions_dir = self.sessions_dir.clone();
        // ponytail: std thread + try_recv keeps the frame loop free without a
        // dedicated async worker type; swap to spawn_blocking if we already hold
        // a Handle in more places.
        std::thread::spawn(move || {
            let _ = tx.send(merged_session_summaries(&sessions_dir));
        });
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn open_history_tree_dialog(&mut self, entries: &[transcript_projection::SessionHistoryEntry]) {
        if entries.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_transcript_entries"));
            return;
        }

        let mut dialog = DialogState::new(
            DialogKind::HistoryTree,
            self.state.t("runtime.session_history"),
            Some(self.state.t("runtime.select_entry")),
            history_tree_dialog_items(entries),
        );
        dialog.selected = dialog.items.len().saturating_sub(1);
        self.state.open_dialog(dialog);
    }

    fn show_thoughts_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::ThoughtsPicker,
            self.state.t("runtime.thinking_display_title"),
            Some(self.state.t("runtime.thinking_display_description")),
            vec![
                DialogItem::new(
                    ThoughtsDisplayMode::Compact.as_str(),
                    self.state.t("runtime.thinking_level_compact"),
                    Some(self.state.t("runtime.thinking_level_compact_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Titles.as_str(),
                    self.state.t("runtime.thinking_level_titles"),
                    Some(self.state.t("runtime.thinking_level_titles_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Scroll.as_str(),
                    self.state.t("runtime.thinking_level_scroll"),
                    Some(self.state.t("runtime.thinking_level_scroll_desc")),
                ),
                DialogItem::new(
                    ThoughtsDisplayMode::Full.as_str(),
                    self.state.t("runtime.thinking_level_full"),
                    Some(self.state.t("runtime.thinking_level_full_desc")),
                ),
            ],
        );
        dialog.selected = match self.state.thoughts_display {
            ThoughtsDisplayMode::Compact => 0,
            ThoughtsDisplayMode::Titles => 1,
            ThoughtsDisplayMode::Scroll => 2,
            ThoughtsDisplayMode::Full => 3,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_tools_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::ToolsPicker,
            self.state.t("runtime.tools_display_title"),
            Some(self.state.t("runtime.tools_display_description")),
            vec![
                DialogItem::new(
                    ToolsDisplayMode::Compact.as_str(),
                    self.state.t("runtime.tools_level_compact"),
                    Some(self.state.t("runtime.tools_level_compact_desc")),
                ),
                DialogItem::new(
                    ToolsDisplayMode::Detailed.as_str(),
                    self.state.t("runtime.tools_level_detailed"),
                    Some(self.state.t("runtime.tools_level_detailed_desc")),
                ),
            ],
        );
        dialog.selected = match self.state.tools_display {
            ToolsDisplayMode::Compact => 0,
            ToolsDisplayMode::Detailed => 1,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_reasoning_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let efforts = self.active_reasoning_efforts();
        if efforts.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_configurable_reasoning"));
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            self.state.t("runtime.reasoning_title"),
            Some(self.state.t("runtime.reasoning_description")),
            reasoning_dialog_items(&efforts),
        );
        dialog.selected =
            reasoning_dialog_selected_index(&efforts, self.current_reasoning_effort());
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_context_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = super::state::context_dialog_items(self.state.active_context());
        if items.is_empty() {
            self.push_command_notice(self.state.t("runtime.no_context_details"));
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(
            DialogKind::ContextPicker,
            self.state.t("runtime.context"),
            None,
            items,
        );
        select_active_context_item(
            &mut dialog,
            self.state.active_context().open_detail.as_ref(),
        );
        self.state.open_dialog(dialog);
        self.state.sync_context_picker_preview();
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_mcp_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.show_mcp_dialog_with_state(String::new(), None);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_mcp_dialog_with_state(&mut self, query: String, selected_server: Option<String>) {
        let description = mcp_discovery_description(self.state.mcp_discovery);
        let mut dialog = DialogState::new(
            DialogKind::McpPicker,
            self.state.t("runtime.mcp_servers"),
            description,
            mcp_dialog_items(
                &self.state.mcp_servers,
                &self.state.mcp_updating,
                self.state.language(),
            ),
        );
        dialog.query = query;
        if let Some(server_name) = selected_server
            && let Some(index) = dialog.items.iter().position(|item| item.id == server_name)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
    }

    fn refresh_open_mcp_dialog(&mut self) {
        let items = mcp_dialog_items(
            &self.state.mcp_servers,
            &self.state.mcp_updating,
            self.state.language(),
        );
        let description = mcp_discovery_description(self.state.mcp_discovery);
        let Some(dialog) = self
            .state
            .dialog_mut()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
        else {
            return;
        };

        let selected_id = dialog
            .items
            .get(dialog.selected)
            .map(|item| item.id.clone());
        dialog.items = items;
        dialog.description = description;
        dialog.selected = selected_id
            .as_deref()
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| dialog.selected.min(dialog.items.len().saturating_sub(1)));
    }

    fn show_skill_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        self.state.open_dialog(DialogState::new(
            DialogKind::SkillPicker,
            self.state.t("ui.skills_title"),
            None,
            skill_dialog_items(&self.state.skill_cards),
        ));
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_mcp_tools_dialog(&mut self, server_name: String) {
        let (primary_query, primary_selected_server) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
            .map(|dialog| {
                (
                    dialog.query.clone(),
                    dialog.selected_item().map(|item| item.id.clone()),
                )
            })
            .unwrap_or_default();
        let tools = self
            .state
            .mcp_server_tools
            .get(&server_name)
            .cloned()
            .unwrap_or_default();
        let description = self.mcp_tools_dialog_description(&server_name);
        let mut dialog = DialogState::new(
            DialogKind::McpToolsPicker,
            self.state
                .t_fmt("runtime.mcp_tools", &[("server", &server_name)]),
            description,
            mcp_tool_dialog_items(&tools),
        );
        dialog.mcp_server_name = Some(server_name);
        dialog.mcp_primary_query = Some(primary_query);
        dialog.mcp_primary_selected_server = primary_selected_server;
        self.state.open_dialog(dialog);
    }

    fn mcp_tools_dialog_description(&self, server_name: &str) -> Option<String> {
        self.state
            .mcp_servers
            .iter()
            .find(|server| server.name == server_name)
            .map(|server| match &server.status {
                mcp::McpServerStatus::Disabled => self.state.t("runtime.mcp_disabled"),
                mcp::McpServerStatus::Online { tool_count } => self
                    .state
                    .t_fmt("runtime.mcp_online", &[("count", &tool_count.to_string())]),
                mcp::McpServerStatus::Offline { .. } => self.state.t("runtime.mcp_offline"),
            })
    }

    fn refresh_open_mcp_tools_dialog(&mut self, server_name: &str) {
        let tools = self
            .state
            .mcp_server_tools
            .get(server_name)
            .cloned()
            .unwrap_or_default();
        let description = self.mcp_tools_dialog_description(server_name);
        let Some(dialog) = self.state.dialog_mut().filter(|dialog| {
            dialog.kind == DialogKind::McpToolsPicker
                && dialog.mcp_server_name.as_deref() == Some(server_name)
        }) else {
            return;
        };

        let selected_id = dialog.selected_item().map(|item| item.id.clone());
        dialog.items = mcp_tool_dialog_items(&tools);
        dialog.description = description;
        dialog.selected = selected_id
            .as_deref()
            .and_then(|id| dialog.items.iter().position(|item| item.id == id))
            .unwrap_or_else(|| dialog.selected.min(dialog.items.len().saturating_sub(1)));
    }

    fn handle_mcp_toggle(&mut self) -> Result<Option<RuntimeCommand>> {
        let Some(server_name) = self
            .state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::McpPicker)
            .and_then(|dialog| dialog.selected_item())
            .map(|item| item.id.clone())
        else {
            return Ok(None);
        };
        if self.state.mcp_updating.contains(&server_name) {
            self.show_toast(
                self.state.t("runtime.mcp_update_progress"),
                ToastKind::Error,
            );
            return Ok(None);
        }
        if !self.has_active_or_pending_session_turn() {
            self.state
                .set_mcp_server_updating(server_name.clone(), true);
            self.refresh_open_mcp_dialog();
        }
        Ok(Some(RuntimeCommand::ToggleMcpServer(server_name)))
    }

    fn handle_dialog_accept(&mut self) -> Result<Option<RuntimeCommand>> {
        let Some((kind, selected)) = self.state.dialog().and_then(|dialog| {
            dialog
                .selected_item()
                .cloned()
                .map(|item| (dialog.kind.clone(), item))
        }) else {
            self.state.close_dialog();
            return Ok(None);
        };

        match kind {
            DialogKind::ModelPicker => {
                self.state.close_dialog();
                self.handle_backend_session_command(crate::session::SessionCommand::SetModel(
                    selected.id,
                ))
                .map(|command| match command {
                    Some(SubmittedCommand::Runtime(command)) => Some(command),
                    Some(SubmittedCommand::LocalOnly) | None => None,
                })
            }
            DialogKind::AgentPicker => {
                self.show_expert_model_dialog(selected.id);
                Ok(None)
            }
            DialogKind::ExpertModelPicker(agent_name) => {
                let (model_ids, query, selected_agent) = self
                    .state
                    .dialog()
                    .map(|dialog| {
                        (
                            dialog
                                .items
                                .iter()
                                .filter(|item| item.checked)
                                .map(|item| item.id.clone())
                                .collect::<Vec<_>>(),
                            dialog.expert_primary_query.clone().unwrap_or_default(),
                            dialog.expert_primary_selected_agent.clone(),
                        )
                    })
                    .unwrap_or_default();
                self.show_agents_dialog_with_state(query, selected_agent);
                Ok(Some(RuntimeCommand::SetExpertAllowedModels {
                    agent_name,
                    model_ids,
                }))
            }
            DialogKind::ConfigEditor => {
                self.handle_config_editor_accept();
                Ok(None)
            }
            DialogKind::PermissionPicker => {
                self.state.close_dialog();
                let mode = match selected.id.as_str() {
                    "safe" => PermissionMode::Safe,
                    "auto" => PermissionMode::Auto,
                    "yolo" => PermissionMode::Yolo,
                    _ => PermissionMode::Default,
                };
                Ok(Some(RuntimeCommand::SetPermissionMode(mode)))
            }
            DialogKind::ReasoningPicker => {
                self.state.close_dialog();
                let effort = parse_reasoning_effort(&selected.id)
                    .expect("reasoning picker items should use valid effort ids");
                if !self.active_reasoning_efforts().contains(&effort) {
                    self.push_command_notice(
                        "That reasoning effort is not supported by the selected model",
                    );
                    return Ok(None);
                }
                Ok(Some(RuntimeCommand::SetReasoningEffort(effort)))
            }
            DialogKind::ThoughtsPicker => {
                self.state.close_dialog();
                let mode = ThoughtsDisplayMode::parse(&selected.id)
                    .expect("thinking display picker items should use valid ids");
                self.apply_thoughts_display(mode);
                Ok(None)
            }
            DialogKind::ToolsPicker => {
                self.state.close_dialog();
                let mode = ToolsDisplayMode::parse(&selected.id)
                    .expect("tools display picker items should use valid ids");
                self.apply_tools_display(mode);
                Ok(None)
            }
            DialogKind::ThemePicker => {
                self.state.close_dialog();
                self.apply_theme_selection(&selected.id);
                Ok(None)
            }
            DialogKind::FakePicker => {
                self.state.close_dialog();
                let client = crate::fake::FakeClient::parse(&selected.id);
                match self.apply_fake_selection(client) {
                    SubmittedCommand::LocalOnly => Ok(None),
                    SubmittedCommand::Runtime(command) => Ok(Some(command)),
                }
            }
            DialogKind::SessionPicker => {
                self.state.close_dialog();
                Ok(Some(RuntimeCommand::ResumeSession(selected.id)))
            }
            DialogKind::HistoryTree => {
                if self.history_navigation_is_unavailable() {
                    self.state.close_dialog();
                    self.state.show_toast(
                        "History navigation is unavailable while work is pending",
                        ToastKind::Info,
                    );
                    return Ok(None);
                }
                self.state.close_dialog();
                let records = read_records(self.sessions_dir.join(format!(
                    "{}.jsonl",
                    self.state.session_id.as_deref().unwrap_or_default()
                )))?;
                let entries = transcript_projection::project_session_history_tree(&records);
                let Some(entry) = entries.into_iter().find(|entry| entry.id == selected.id) else {
                    return Ok(None);
                };
                let target_id =
                    if entry.kind == transcript_projection::SessionHistoryEntryKind::User {
                        entry.parent_id.clone().or_else(|| Some("entry-0".into()))
                    } else {
                        Some(entry.id.clone())
                    };
                if let Some(content) = entry.user_content {
                    self.state.set_composer_content(content);
                }
                Ok(target_id
                    .map(|target_entry_id| RuntimeCommand::NavigateHistory { target_entry_id }))
            }
            DialogKind::ContextPicker => {
                let detail_focused = self
                    .state
                    .dialog()
                    .is_some_and(|dialog| dialog.detail_focused);
                if !detail_focused
                    && self.state.active_context_open_detail().is_some()
                    && let Some(dialog) = self.state.dialog_mut()
                {
                    dialog.detail_focused = true;
                    dialog.detail_scroll = 0;
                }
                Ok(None)
            }
            DialogKind::LanguagePicker => {
                let language = match selected.id.as_str() {
                    "zh-CN" => crate::tui::i18n::Language::ZhCn,
                    _ => crate::tui::i18n::Language::En,
                };
                self.state.set_language(Some(language));
                TuiPreferences::update_in_dir(&self.preferences_dir, |prefs| {
                    prefs.language = Some(language.id().to_string());
                })?;
                self.state.close_dialog();
                Ok(None)
            }
            DialogKind::SkillPicker => {
                let attached = self.state.add_composer_skill(selected.id);
                self.state.close_dialog();
                if !attached {
                    self.show_toast("Skill already attached", ToastKind::Info);
                }
                Ok(None)
            }
            DialogKind::McpPicker => {
                self.show_mcp_tools_dialog(selected.id);
                Ok(None)
            }
            DialogKind::McpToolsPicker => Ok(None),
            DialogKind::ContextDetail => {
                self.state.close_dialog();
                Ok(None)
            }
        }
    }

    fn notify_context_dialog_issue(&mut self, summary: &str, detail: &str) {
        self.show_toast(summary, ToastKind::Error);
        tracing::warn!(%summary, %detail, "context dialog issue");
    }

    fn sync_context_inspector_preview(&mut self) {
        let Some(dialog) = self.state.dialog() else {
            return;
        };
        if dialog.kind != DialogKind::ContextPicker {
            return;
        }

        let selected_id = dialog.selected_item().map(|item| item.id.clone());
        let Some(selected_id) = selected_id else {
            self.state.open_context_detail(None);
            return;
        };

        let Some(target) = parse_context_dialog_target(&selected_id) else {
            self.state.open_context_detail(None);
            self.notify_context_dialog_issue(
                &self.state.t("runtime.context_item_unavailable"),
                "Refresh context and try again",
            );
            return;
        };

        if !context_detail_available(self.state.active_context(), &target) {
            self.state.open_context_detail(None);
            self.notify_context_dialog_issue(
                &self.state.t("runtime.context_item_unavailable"),
                "Refresh context and try again",
            );
            return;
        }

        self.state.open_context_detail(Some(target));
    }

    fn set_permission_mode_command(&mut self, mode: PermissionMode) -> SubmittedCommand {
        SubmittedCommand::Runtime(RuntimeCommand::SetPermissionMode(mode))
    }

    fn set_reasoning_effort_command(&mut self, effort: ModelReasoningEffort) -> SubmittedCommand {
        if !self.active_reasoning_efforts().contains(&effort) {
            self.push_command_notice(
                "That reasoning effort is not supported by the selected model",
            );
            return SubmittedCommand::LocalOnly;
        }
        SubmittedCommand::Runtime(RuntimeCommand::SetReasoningEffort(effort))
    }

    fn cycle_reasoning_effort_command(&mut self) -> Option<RuntimeCommand> {
        let efforts = self.active_reasoning_efforts();
        let Some(next) = next_reasoning_effort(&efforts, self.current_reasoning_effort()) else {
            self.push_command_notice(self.state.t("runtime.no_configurable_reasoning"));
            return None;
        };
        Some(RuntimeCommand::SetReasoningEffort(next))
    }

    fn active_reasoning_efforts(&self) -> Vec<ModelReasoningEffort> {
        self.available_models
            .iter()
            .find(|model| model.id == self.state.model_id)
            .map(|model| model.reasoning_efforts.clone())
            .unwrap_or_default()
    }

    fn current_reasoning_effort(&self) -> Option<ModelReasoningEffort> {
        match parse_reasoning_effort(
            self.state
                .pending_composer_settings
                .reasoning_effort
                .as_deref()
                .or(self.state.reasoning_effort_label.as_deref())
                .unwrap_or("off"),
        ) {
            Some(ModelReasoningEffort::None) | None => None,
            Some(effort) => Some(effort),
        }
    }

    fn push_command_notice(&mut self, message: impl Into<String>) {
        self.state.show_toast(message.into(), ToastKind::Info);
    }

    fn selected_slash_command(&self) -> Option<SlashCommandEntry> {
        let matches = matching_completion_commands(&self.state.input_buffer);
        matches
            .get(
                self.state
                    .slash_panel_selected
                    .min(matches.len().saturating_sub(1)),
            )
            .copied()
    }

    fn select_next_slash_command(&mut self) {
        let matches = matching_completion_commands(&self.state.input_buffer);
        if matches.is_empty() {
            self.state.slash_panel_selected = 0;
            return;
        }

        self.state.slash_panel_selected = (self.state.slash_panel_selected + 1) % matches.len();
    }

    fn select_previous_slash_command(&mut self) {
        let matches = matching_completion_commands(&self.state.input_buffer);
        if matches.is_empty() {
            self.state.slash_panel_selected = 0;
            return;
        }

        self.state.slash_panel_selected = if self.state.slash_panel_selected == 0 {
            matches.len().saturating_sub(1)
        } else {
            self.state.slash_panel_selected.saturating_sub(1)
        };
    }

    fn accept_selected_slash_command(&mut self) {
        if let Some(selected) = self.selected_slash_command() {
            self.state.set_input(selected.insert_text);
        }
    }

    fn handle_transcript_click(&mut self, col: u16, row: u16, activate_link: bool) {
        match self.state.transcript_click_target(col, row) {
            Some(TranscriptClickTarget::OpenUrl(url)) if activate_link => {
                if let Err(error) = super::transcript_ratatui::open_hyperlink_url(&url) {
                    self.show_toast(
                        self.state
                            .t_fmt("runtime.failed_open_link", &[("error", &error.to_string())]),
                        ToastKind::Error,
                    );
                }
            }
            Some(TranscriptClickTarget::OpenUrl(_)) => {}
            Some(TranscriptClickTarget::ToolCard(call_id)) if !activate_link => {
                self.state.toggle_tool_output(&call_id);
            }
            Some(TranscriptClickTarget::ToolCard(_)) | None => {}
        }
    }

    fn handle_scrollbar_drag_start(&mut self, column: u16, row: u16) {
        let area = self.state.last_scrollbar_area;
        if column < area.x || column >= area.right() || row < area.y || row >= area.bottom() {
            return;
        }
        let track_row = usize::from(row - area.y);
        let Some(geometry) = self.state.transcript_scrollbar() else {
            return;
        };
        if !geometry.contains_row(track_row) {
            return;
        }
        self.state.transcript_scrollbar_drag = Some(geometry.grab_offset(track_row));
    }

    fn handle_scrollbar_drag_move(&mut self, row: u16) {
        let Some(grab) = self.state.transcript_scrollbar_drag else {
            return;
        };
        let area = self.state.last_scrollbar_area;
        let Some(geometry) = self.state.transcript_scrollbar() else {
            return;
        };
        let track_row = usize::from(row.saturating_sub(area.y));
        let position = geometry.position_for_grab(track_row, grab);
        self.state.scroll_transcript_to_top_row(position);
    }

    fn handle_selection_start(&mut self, col: u16, row: u16) {
        // 落在 transcript 内容区外不开始选择；点击空白/spacer 也返回 None
        if let Some(anchor) = self.state.map_mouse_to_anchor(col, row) {
            self.state.text_selection = Some(super::state::TextSelection {
                start: anchor.clone(),
                end: anchor,
            });
            self.state.selection_in_progress = true;
            self.state.selection_dragged = false;
            self.state.selection_last_mouse = Some((col, row));
        } else {
            // 在 transcript 外点击：清除现有选择，避免残留高亮
            self.state.text_selection = None;
            self.state.selection_in_progress = false;
            self.state.selection_dragged = false;
            self.state.selection_last_mouse = None;
        }
    }

    fn handle_selection_drag(&mut self, col: u16, row: u16) {
        if !self.state.selection_in_progress {
            return;
        }
        self.state.selection_last_mouse = Some((col, row));
        if let Some(anchor) = self.state.map_mouse_to_anchor(col, row) {
            let anchor_changed = self
                .state
                .text_selection
                .as_ref()
                .is_some_and(|selection| selection.end != anchor);
            if anchor_changed {
                self.state.selection_dragged = true;
                if let Some(selection) = &mut self.state.text_selection {
                    selection.end = anchor;
                }
            }
        }
    }

    fn handle_selection_end(&mut self, col: u16, row: u16, activate_link: bool) {
        let dragged = self.state.selection_dragged;
        if dragged {
            self.handle_selection_drag(col, row);
        }
        self.state.selection_in_progress = false;
        self.state.selection_dragged = false;
        self.state.selection_last_mouse = None;
        // 抛弃零宽选择（单击未拖动），避免接管 Ctrl+C 复制语义且无视觉反馈
        if let Some(selection) = &self.state.text_selection
            && selection.start == selection.end
        {
            self.state.text_selection = None;
        }
        if !dragged {
            self.handle_transcript_click(col, row, activate_link);
        }
    }

    /// 拖拽选择期间，鼠标停留在 transcript 顶/底边缘时自动滚动并扩展选择终点。
    /// 在每帧 Tick 调用一次，约 30fps。
    fn tick_selection_autoscroll(&mut self) {
        if !self.state.selection_in_progress {
            return;
        }
        let Some((col, row)) = self.state.selection_last_mouse else {
            return;
        };
        let area = self.state.last_transcript_area;
        if area.height == 0 {
            return;
        }
        // 边缘触发带：顶部/底部 2 行内。鼠标被拖到 area 之外（row < top 或 >= bottom）
        // 也视为边缘，以便继续选择刚被滚动露出的内容。
        const EDGE_BAND: u16 = 2;
        let band = EDGE_BAND.min(area.height);

        let scrolled_up = row < area.top() + band;
        let scrolled_down = row >= area.bottom().saturating_sub(band);
        if scrolled_up {
            self.state.scroll_transcript_up(1);
        } else if scrolled_down {
            self.state.scroll_transcript_down(1);
        } else {
            return;
        }

        // 鼠标可能已位于 area 之外（命中检测会失败），为让选择继续扩展到刚滚出的行，
        // 用 clamp 到 area 边界的列/行来映射 selection.end。
        let clamped_col = col.clamp(area.left(), area.right().saturating_sub(1));
        let clamped_row = if scrolled_up {
            area.top()
        } else {
            area.bottom().saturating_sub(1)
        };
        if let Some(anchor) = self.state.map_mouse_to_anchor(clamped_col, clamped_row)
            && let Some(selection) = &mut self.state.text_selection
        {
            selection.end = anchor;
        }
    }

    fn handle_copy_selection(&mut self) -> Result<()> {
        let text = crate::tui::selection::extract_selected_text(&self.state);
        if !text.is_empty() {
            self.copy_to_clipboard(text, "runtime.copied_clipboard");
        }
        Ok(())
    }

    fn handle_copy_session_id(&mut self) {
        let Some(session_id) = self.state.session_id.clone() else {
            return;
        };
        self.copy_to_clipboard(session_id, "runtime.copied_session_id");
    }

    fn copy_to_clipboard(&mut self, text: String, copied_key: &str) {
        use arboard::Clipboard;

        match Clipboard::new() {
            Ok(mut clipboard) => {
                if clipboard.set_text(text).is_err() {
                    self.show_toast(self.state.t("runtime.copy_failed"), ToastKind::Error);
                } else {
                    self.show_toast(self.state.t(copied_key), ToastKind::Success);
                }
            }
            Err(_) => {
                self.show_toast(
                    self.state.t("runtime.clipboard_unavailable"),
                    ToastKind::Error,
                );
            }
        }
    }

    fn clipboard_paste_context(&self) -> ClipboardPasteContext {
        if self.state.dialog_is_open() {
            ClipboardPasteContext::Dialog
        } else if self.state.pending_question.is_some() {
            ClipboardPasteContext::Question
        } else if self.state.pending_permission.is_some() {
            ClipboardPasteContext::Permission
        } else {
            ClipboardPasteContext::Composer
        }
    }

    fn handle_paste_from_clipboard(&mut self) -> Result<()> {
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                let text = clipboard.get_text().ok().filter(|text| !text.is_empty());
                let composer = matches!(
                    self.clipboard_paste_context(),
                    ClipboardPasteContext::Composer
                );
                let image = if composer {
                    clipboard.get_image().ok()
                } else {
                    None
                };
                let image_files = if composer && image.is_none() {
                    clipboard.get().file_list().unwrap_or_default()
                } else {
                    Vec::new()
                };
                if let Err(error) = self.apply_clipboard_content(text, image, &image_files) {
                    tracing::warn!(%error, "failed to paste clipboard content");
                    self.show_toast(self.state.t("runtime.paste_failed"), ToastKind::Error);
                }
            }
            Err(error) => {
                tracing::warn!(%error, "failed to open clipboard");
                self.show_toast(
                    self.state.t("runtime.clipboard_unavailable"),
                    ToastKind::Error,
                );
            }
        }
        Ok(())
    }

    fn apply_clipboard_content(
        &mut self,
        text: Option<String>,
        image: Option<arboard::ImageData<'_>>,
        image_files: &[PathBuf],
    ) -> Result<()> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

        if image.is_none()
            && matches!(
                self.clipboard_paste_context(),
                ClipboardPasteContext::Composer
            )
        {
            let attachments = clipboard_image_attachments(image_files);
            if !attachments.is_empty() {
                if self.state.is_read_only_child_view() {
                    return Ok(());
                }
                for attachment in attachments {
                    self.state.add_composer_attachment(attachment);
                }
                self.reset_history_navigation();
                return Ok(());
            }
        }

        match choose_clipboard_paste(
            self.clipboard_paste_context(),
            text.is_some(),
            image.is_some(),
        ) {
            ClipboardPasteChoice::Image => {
                if self.state.is_read_only_child_view() {
                    return Ok(());
                }
                let image = image.expect("clipboard image choice requires an image");
                let width = u32::try_from(image.width)?;
                let height = u32::try_from(image.height)?;
                anyhow::ensure!(
                    width > 0
                        && height > 0
                        && image
                            .width
                            .checked_mul(image.height)
                            .and_then(|pixels| pixels.checked_mul(4))
                            == Some(image.bytes.len()),
                    "invalid clipboard RGBA image dimensions"
                );
                let mut png_bytes = Vec::new();
                PngEncoder::new(&mut png_bytes).write_image(
                    image.bytes.as_ref(),
                    width,
                    height,
                    ColorType::Rgba8.into(),
                )?;
                let data_url = format!("data:image/png;base64,{}", STANDARD.encode(png_bytes));
                self.state.add_composer_attachment(UserImageAttachment {
                    id: next_attachment_id(),
                    label: "clipboard".into(),
                    mime: "image/png".into(),
                    data_url,
                });
                self.reset_history_navigation();
            }
            ClipboardPasteChoice::Text => {
                let action = map_paste_event(
                    &self.state,
                    text.expect("clipboard text choice requires text"),
                );
                let _ = self.handle_input_action(action)?;
            }
            ClipboardPasteChoice::None => {
                self.show_toast(self.state.t("runtime.paste_failed"), ToastKind::Error);
            }
        }
        Ok(())
    }
}

fn composer_draft_for_submission(state: &TuiState) -> ComposerDraft {
    let mut input_buffer = state.input_buffer.trim().to_string();
    let mut tokens = state.composer_tokens.clone();
    if input_buffer.starts_with(crate::tui::state::COMPOSER_ATTACHMENT_MARKER)
        && let Some(crate::tui::state::ComposerToken::PastedText(text)) = tokens.first_mut()
    {
        *text = text.trim_start().to_string();
    }
    if input_buffer.ends_with(crate::tui::state::COMPOSER_ATTACHMENT_MARKER)
        && let Some(crate::tui::state::ComposerToken::PastedText(text)) = tokens.last_mut()
    {
        *text = text.trim_end().to_string();
    }
    if input_buffer.is_empty() {
        tokens.clear();
    }
    let input_cursor = input_buffer.len();
    ComposerDraft {
        input_buffer: std::mem::take(&mut input_buffer),
        input_cursor,
        tokens,
    }
}

fn parse_reasoning_effort(value: &str) -> Option<ModelReasoningEffort> {
    crate::command::parse_reasoning_effort(value)
}

fn reasoning_effort_config_label(effort: &ModelReasoningEffort) -> &str {
    effort.as_str()
}

fn reasoning_effort_status_label(effort: Option<ModelReasoningEffort>) -> String {
    match effort {
        Some(ModelReasoningEffort::None) | None => "off".into(),
        Some(effort) => reasoning_effort_config_label(&effort).into(),
    }
}

fn next_reasoning_effort(
    efforts: &[ModelReasoningEffort],
    current: Option<ModelReasoningEffort>,
) -> Option<ModelReasoningEffort> {
    if efforts.is_empty() {
        return None;
    }

    let current = current.unwrap_or(ModelReasoningEffort::None);
    let index = efforts
        .iter()
        .position(|effort| *effort == current)
        .map(|index| (index + 1) % efforts.len())
        .unwrap_or(0);
    efforts.get(index).cloned()
}

fn reasoning_dialog_items(efforts: &[ModelReasoningEffort]) -> Vec<DialogItem> {
    efforts
        .iter()
        .map(|effort| {
            let (label, detail) = match effort {
                ModelReasoningEffort::None => ("Off", "Do not request extra reasoning"),
                ModelReasoningEffort::Minimal => ("Minimal", "Smallest reasoning budget"),
                ModelReasoningEffort::Low => ("Low", "Light reasoning budget"),
                ModelReasoningEffort::Medium => ("Medium", "Balanced reasoning budget"),
                ModelReasoningEffort::High => ("High", "Deeper reasoning budget"),
                ModelReasoningEffort::Xhigh => ("XHigh", "Very deep reasoning budget"),
                ModelReasoningEffort::Max => ("Max", "Provider-specific maximum reasoning budget"),
                ModelReasoningEffort::Custom(_) => {
                    (effort.as_str(), "Provider-specific reasoning budget")
                }
            };
            DialogItem::new(
                reasoning_effort_config_label(effort),
                label,
                Some(detail.into()),
            )
        })
        .collect()
}

fn reasoning_dialog_selected_index(
    efforts: &[ModelReasoningEffort],
    current: Option<ModelReasoningEffort>,
) -> usize {
    let current = current.unwrap_or(ModelReasoningEffort::None);
    efforts
        .iter()
        .position(|effort| *effort == current)
        .unwrap_or(0)
}

fn child_view_allows_prompt(prompt: &str) -> bool {
    let prompt = prompt.trim();
    if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
        return true;
    }

    if !prompt.starts_with('/') {
        return false;
    }

    let Some(name) = prompt.split_whitespace().next() else {
        return false;
    };

    matches!(
        name,
        "/help"
            | "/?"
            | "/exit"
            | "/quit"
            | "/child"
            | "/thoughts"
            | "/tools"
            | "/tool-output"
            | "/scrollbar"
            | "/theme"
            | "/context"
    )
}

fn custom_theme_description(state: &TuiState, custom: &CustomThemeInfo) -> Option<String> {
    let key = match custom.id.as_str() {
        "ocean" => "runtime.theme_ocean_desc",
        "forest" => "runtime.theme_forest_desc",
        "rose" => "runtime.theme_rose_desc",
        "tokyonight" => "runtime.theme_tokyonight_desc",
        _ => return custom.description.clone(),
    };
    match bundled_theme_description(&custom.id) {
        Some(default) if custom.description.as_deref() == Some(default.as_str()) => {
            Some(state.t(key))
        }
        _ => custom.description.clone(),
    }
}

#[cfg(test)]
fn context_dialog_items(context: &super::state::ContextPaneState) -> Vec<DialogItem> {
    let mut items = Vec::new();

    for node in context.tree.nodes() {
        if node.node_id == *context.tree.root_node_id() {
            continue;
        }
        let depth = context_node_depth(&context.tree, node.node_id.as_str());
        let indent = if depth == 0 {
            String::new()
        } else {
            format!("{}↳ ", "  ".repeat(depth.saturating_sub(1)))
        };
        let mut label = format!(
            "{indent}{}",
            node.label
                .clone()
                .unwrap_or_else(|| node.node_id.as_str().to_string())
        );
        if context.tree.active_node_id() == Some(&node.node_id) {
            label.push_str(" · Active");
        }
        if node.status == crate::context_tree::ContextNodeStatus::Archived {
            label.push_str(" · Archived");
        }
        items.push(
            DialogItem::new(
                format!("node:{}", node.node_id.as_str()),
                label,
                node.purpose.clone(),
            )
            .with_section("Nodes"),
        );
    }

    for (_, block) in context.view.provider_active_blocks() {
        let mut detail = context_block_status_labels(&context.view, block).join(" · ");
        if detail.is_empty() {
            detail = block_source_label(block).to_string();
        }
        items.push(
            DialogItem::new(
                format!("block:{}", block.block_id.as_str()),
                block.title.clone(),
                Some(detail),
            )
            .with_section("Blocks"),
        );
    }

    for artifact in context.view.summary_artifacts.iter().filter(|artifact| {
        context.view.provider_active_blocks().iter().any(|(_, block)| {
            matches!(&block.source, crate::context_view::ContextBlockSource::SummaryArtifact { artifact_id }
                if artifact_id == &artifact.artifact_id)
        })
    }) {
        items.push(
            DialogItem::new(
                format!("summary:{}", artifact.artifact_id),
                format!("Summary {}", artifact.artifact_id),
                Some(artifact.node_id.clone()),
            )
            .with_section("Summaries"),
        );
    }

    items
}

fn parse_context_dialog_target(id: &str) -> Option<ContextDetailTarget> {
    let (kind, value) = id.split_once(':')?;
    match kind {
        "node" => Some(ContextDetailTarget::Node(value.to_string())),
        "block" => Some(ContextDetailTarget::Block(value.to_string())),
        "summary" => Some(ContextDetailTarget::Summary(value.to_string())),
        _ => None,
    }
}

fn context_dialog_target_id(target: &ContextDetailTarget) -> String {
    match target {
        ContextDetailTarget::Node(node_id) => format!("node:{node_id}"),
        ContextDetailTarget::Block(block_id) => format!("block:{block_id}"),
        ContextDetailTarget::Summary(artifact_id) => format!("summary:{artifact_id}"),
    }
}

fn select_active_context_item(dialog: &mut DialogState, target: Option<&ContextDetailTarget>) {
    let Some(target) = target else {
        return;
    };
    let target_id = context_dialog_target_id(target);
    if let Some(index) = dialog.items.iter().position(|item| item.id == target_id) {
        dialog.selected = index;
    }
}

fn context_detail_available(
    context: &super::state::ContextPaneState,
    target: &ContextDetailTarget,
) -> bool {
    super::state::context_detail_target_exists(context, target)
}

#[cfg(test)]
fn context_detail_dialog(
    context: &super::state::ContextPaneState,
    target: &ContextDetailTarget,
) -> Option<DialogState> {
    if !context_detail_available(context, target) {
        return None;
    }
    let (title, lines) = match target {
        ContextDetailTarget::Node(node_id) => {
            let node = context
                .tree
                .nodes()
                .find(|node| node.node_id.as_str() == node_id)?;
            let title = node
                .label
                .clone()
                .unwrap_or_else(|| node.node_id.as_str().to_string());
            let mut lines = Vec::new();
            lines.push(DialogItem::new(
                "status",
                "Status",
                Some(format!("{:?}", node.status)),
            ));
            if let Some(purpose) = node.purpose.clone() {
                lines.push(DialogItem::new("purpose", "Purpose", Some(purpose)));
            }
            if let Some(source_ref) = node.source_ref.as_ref() {
                lines.push(DialogItem::new(
                    "source",
                    "Source",
                    Some(match source_ref.source_id.as_deref() {
                        Some(source_id) => format!("{}:{}", source_ref.source_kind, source_id),
                        None => source_ref.source_kind.clone(),
                    }),
                ));
            }
            (title, lines)
        }
        ContextDetailTarget::Block(block_id) => {
            let block = context
                .view
                .blocks
                .iter()
                .find(|(candidate, _)| candidate.as_str() == block_id)
                .map(|(_, block)| block)?;
            if context.view.is_compacted(&block.block_id) {
                return None;
            }
            if context.view.view_state.status(&block.block_id)
                == Some(crate::context_view::ContextViewStatus::RemovedFromView)
            {
                return None;
            }
            let mut lines = vec![DialogItem::new(
                "status",
                "Status",
                Some(context_block_status_labels(&context.view, block).join(" · ")),
            )];
            lines.push(DialogItem::new(
                "detail",
                "Open detail",
                Some(truncate_dialog_text(&block.detail)),
            ));
            for source in context_block_detail_lines(block, &context.view) {
                lines.push(DialogItem::new("source", source.0, Some(source.1)));
            }
            (block.title.clone(), lines)
        }
        ContextDetailTarget::Summary(artifact_id) => {
            let artifact = context.view.open_summary_artifact(artifact_id)?;
            let mut lines = vec![DialogItem::new(
                "summary",
                "Open detail",
                Some(truncate_dialog_text(&artifact.summary)),
            )];
            if let Some(node_id) = artifact.source_node_id.clone() {
                lines.push(DialogItem::new("node", "Source", Some(node_id)));
            }
            if let Some(block_id) = artifact.source_block_id.clone() {
                lines.push(DialogItem::new("block", "Block", Some(block_id)));
            }
            (format!("Summary {}", artifact.artifact_id), lines)
        }
    };

    Some(DialogState::new(
        DialogKind::ContextDetail,
        format!("Detail · {title}"),
        None,
        lines,
    ))
}

#[cfg(test)]
fn context_node_depth(tree: &crate::context_tree::ContextTreeState, node_id: &str) -> usize {
    let mut depth = 0usize;
    let mut current = tree
        .nodes()
        .find(|node| node.node_id.as_str() == node_id)
        .and_then(|node| node.parent_node_id.clone());
    while let Some(parent) = current {
        if parent == *tree.root_node_id() {
            break;
        }
        depth = depth.saturating_add(1);
        current = tree
            .node(&parent)
            .and_then(|node| node.parent_node_id.clone());
    }
    depth
}

#[cfg(test)]
fn context_block_status_labels(
    view: &crate::context_view::ContextViewProjection,
    block: &crate::context_view::ContextBlock,
) -> Vec<String> {
    let mut labels = Vec::new();
    match view.view_state.status(&block.block_id) {
        Some(crate::context_view::ContextViewStatus::Pinned) => labels.push("Pinned".into()),
        Some(crate::context_view::ContextViewStatus::Archived) => labels.push("Archived".into()),
        Some(crate::context_view::ContextViewStatus::Resolved) => labels.push("Resolved".into()),
        Some(crate::context_view::ContextViewStatus::RemovedFromView) => {}
        _ => {}
    }
    if matches!(
        block.source,
        crate::context_view::ContextBlockSource::SummaryArtifact { .. }
    ) {
        labels.push("Summary".into());
    }
    if block.is_protected() {
        labels.push("Protected".into());
    }
    labels
}

#[cfg(test)]
fn block_source_label(block: &crate::context_view::ContextBlock) -> &'static str {
    match block.source {
        crate::context_view::ContextBlockSource::TranscriptSpan { .. } => "Source",
        crate::context_view::ContextBlockSource::SummaryArtifact { .. } => "Summary",
    }
}

#[cfg(test)]
fn context_block_detail_lines(
    block: &crate::context_view::ContextBlock,
    view: &crate::context_view::ContextViewProjection,
) -> Vec<(&'static str, String)> {
    let mut lines = Vec::new();
    match &block.source {
        crate::context_view::ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => lines.push(("Source", format!("@{}–@{}", start_sequence, end_sequence))),
        crate::context_view::ContextBlockSource::SummaryArtifact { artifact_id } => {
            lines.push(("Source", format!("Summary {artifact_id}")));
            if let Some(artifact) = view.open_summary_artifact(artifact_id) {
                if let Some(node_id) = artifact.source_node_id.clone() {
                    lines.push(("Node", node_id));
                }
                if let Some(block_id) = artifact.source_block_id.clone() {
                    lines.push(("Block", block_id));
                }
            }
        }
    }
    lines
}

#[cfg(test)]
fn truncate_dialog_text(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= 120 {
        return collapsed;
    }
    let mut out = collapsed.chars().take(120).collect::<String>();
    out.push('…');
    out
}

fn apply_preferences_theme(state: &mut TuiState, preferences_dir: &Path, theme_id: &str) {
    let Some(id) = normalize_theme_id(theme_id) else {
        state.set_theme_name(ThemeName::Dark);
        return;
    };
    if let Some(builtin) = ThemeName::parse(&id) {
        state.set_theme_name(builtin);
        return;
    }
    ensure_bundled_themes(preferences_dir);
    match load_custom_theme(preferences_dir, &id) {
        Ok(palette) => state.set_active_theme(id, Some(palette)),
        Err(error) => {
            tracing::warn!(%error, theme = %id, "failed to load preferred custom theme");
            state.set_theme_name(ThemeName::Dark);
            state.show_toast(
                format!("Theme '{id}' unavailable; using dark"),
                ToastKind::Info,
            );
        }
    }
}

/// A panic inside the alternate screen is wiped when the terminal restores.
fn install_panic_log(dir: &Path) {
    let path = dir.join("panic.log");
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let location = info
                .location()
                .map(|location| format!("{}:{}", location.file(), location.line()))
                .unwrap_or_else(|| "unknown location".to_string());
            let payload = info
                .payload()
                .downcast_ref::<&str>()
                .map(|payload| (*payload).to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic".to_string());
            let _ = writeln!(file, "{location}: {payload}");
        }
        previous(info);
    }));
}

pub async fn run_tui(
    mut engine: SessionEngine,
    projection: crate::session::SessionEngineProjection,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    config_path: PathBuf,
    workspace_dir: PathBuf,
    provider_label: String,
    available_models: Vec<AvailableModel>,
    available_experts: Vec<AvailableExpert>,
    startup_toast: Option<StartupToast>,
    skill_cards: Vec<SkillCard>,
    resume_session_id: Option<String>,
) -> Result<()> {
    install_panic_log(&preferences_dir);
    let mut state = TuiState::new(
        projection.model_id,
        projection.model_label,
        projection.permission_mode_label,
    );
    state.session_id = Some(projection.session_id);
    state.set_skill_cards(skill_cards);
    let preferences = TuiPreferences::load_from_dir(&preferences_dir);
    state.set_language(preferences.explicit_language());
    state.set_tool_output_expanded(preferences.tool_output_expanded);
    state.set_transcript_scrollbar_visible(preferences.transcript_scrollbar_visible);
    state.set_sidebar_preference(preferences.sidebar_hidden, preferences.sidebar_forced_open);
    state.set_thoughts_display(preferences.thoughts_display);
    state.set_tools_display(preferences.tools_display);
    apply_preferences_theme(&mut state, &preferences_dir, &preferences.theme);
    state.set_fake_client(None);
    state.fake_installation_id = preferences.fake_installation_id;
    state.set_provider_label(provider_label);
    state.set_fast_mode_enabled(projection.fast_mode_enabled);

    if let Some(active_model) = available_models
        .iter()
        .find(|model| model.id == state.model_id)
    {
        state.set_model(active_model.id.clone(), active_model.label.clone());
        state.set_model_context_window(active_model.context_window_tokens);
        state.set_reasoning_effort_label(Some(reasoning_effort_status_label(
            active_model.reasoning_effort.clone(),
        )));
    }
    if !projection.api_key_configured {
        let message = state.t("runtime.missing_api_key");
        state.show_toast(message, ToastKind::Info);
    }
    if let Some(toast) = startup_toast {
        state.show_toast(toast.message, toast.kind);
    }
    let ingress = engine.take_ingress();
    let session_transport_rx = engine.take_event_egress().into_receiver();
    let mut exit_epilogue = None;
    let tui_result = async {
        let mut runtime = TuiRuntime::new(
            state,
            session_transport_rx,
            available_models,
            available_experts,
            sessions_dir,
            preferences_dir,
        );
        runtime.set_config_path(config_path);
        runtime.set_workspace_dir(workspace_dir);
        runtime.start_update_check();
        runtime.start_session_archive_pass();
        runtime.session_title = projection.session_title;
        let mut terminal = OwnedTerminal::new()?;
        // 必须在输入读取开始前：探针窗口里读到的按键无法归还事件流。
        runtime
            .state
            .set_terminal_bg(super::terminal_bg::query_background());
        // Restore platform input modes before OwnedTerminal restores raw mode.
        let mut input = super::terminal_input::TerminalInput::new()?;
        runtime.update_terminal_title(&mut terminal)?;
        let mut drawer = TerminalDrawer::new(&mut terminal);

        if let Some(session_id) = resume_session_id {
            runtime.session_resume_pending = true;
            let message = runtime.state.t("runtime.resuming_session");
            runtime.state.show_toast(message, ToastKind::Info);
            command_dispatch::dispatch_command(
                &mut runtime,
                RuntimeCommand::ResumeSession(session_id),
                &ingress,
                true,
            );
        }

        loop {
            for _ in 0..MAX_INPUT_EVENTS_PER_FRAME {
                let Some(event) = input.read(Duration::ZERO)? else {
                    break;
                };
                process_terminal_event(&mut runtime, event, &ingress)?;
            }
            runtime.try_drain_session_events();
            if let Some(command) = runtime.take_next_queued_prompt_command() {
                command_dispatch::dispatch_command(&mut runtime, command, &ingress, true);
            }
            drawer.set_title(&runtime.terminal_title())?;
            runtime.draw(&mut drawer)?;

            if runtime.state().quit_requested {
                if let Some(session_id) = runtime.state().session_id.as_deref() {
                    exit_epilogue = Some(super::render::format_exit_epilogue(
                        session_id,
                        runtime.session_title.as_deref(),
                    ));
                }
                break;
            }
            if let Some(event) = input.read(TUI_FRAME_POLL_INTERVAL)? {
                process_terminal_event(&mut runtime, event, &ingress)?;
            } else {
                let _ = runtime.handle_input_action(InputAction::Tick)?;
            }
        }
        Ok(())
    }
    .await;

    // Terminal has left the alternate screen; print into normal scrollback.
    if let Some(epilogue) = exit_epilogue {
        println!("{epilogue}");
    }

    let shutdown_result = ingress.shutdown().map_err(Into::into);
    drop(ingress);
    let join_result = engine.join().await;
    tui_result.and(shutdown_result).and(join_result)
}

struct TerminalDrawer<'a> {
    terminal: &'a mut OwnedTerminal,
    applied_hyperlink_cells: Vec<super::transcript_ratatui::HyperlinkCell>,
}

impl<'a> TerminalDrawer<'a> {
    fn new(terminal: &'a mut OwnedTerminal) -> Self {
        Self {
            terminal,
            applied_hyperlink_cells: Vec::new(),
        }
    }

    fn set_title(&mut self, title: &str) -> io::Result<()> {
        self.terminal.set_title(title)
    }
}

impl RuntimeDrawer for TerminalDrawer<'_> {
    fn draw(&mut self, state: &mut TuiState) -> io::Result<()> {
        // Ratatui keeps the hardware cursor hidden for frames without a requested
        // cursor. Avoid per-frame cursor moves here: some CJK IMEs follow the VT
        // cursor even while hidden, which made their candidate window jitter.
        let terminal = self.terminal.terminal_mut();
        let completed = terminal.draw(|frame| render::render(frame, state))?;
        let overlay = super::transcript_ratatui::plan_hyperlink_overlay(
            completed.buffer,
            &self.applied_hyperlink_cells,
            &state.frame_hyperlink_cells,
        );
        super::transcript_ratatui::write_hyperlink_overlay(terminal.backend_mut(), &overlay)?;
        self.applied_hyperlink_cells = overlay.applied;
        Ok(())
    }
}

#[cfg(test)]
mod git_branch_tests {
    use super::TuiRuntime;
    use super::branch_poller::read_git_branch;
    use crate::tui::TuiState;
    use crate::tui::state::ToastKind;
    use std::path::Path;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-git-branch-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn git(path: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = mpsc::unbounded_channel();
        TuiRuntime::new(
            TuiState::default(),
            rx,
            Vec::new(),
            Vec::new(),
            temp_dir("sessions"),
            temp_dir("preferences"),
        )
    }

    fn deliver_branch_refresh(runtime: &mut TuiRuntime, branch: Option<String>) {
        runtime.branch_poller.enqueue_for_test(branch);
        runtime.poll_git_branch();
    }

    #[test]
    fn branch_refresh_replaces_and_clears_cached_state() {
        let mut runtime = runtime();

        deliver_branch_refresh(&mut runtime, Some("main".into()));
        assert_eq!(runtime.state().git_branch.as_deref(), Some("main"));

        deliver_branch_refresh(&mut runtime, Some("detached@abc1234".into()));
        assert_eq!(
            runtime.state().git_branch.as_deref(),
            Some("detached@abc1234")
        );

        deliver_branch_refresh(&mut runtime, None);
        assert_eq!(runtime.state().git_branch, None);
    }

    #[test]
    fn update_check_result_shows_localized_info_toast_once() {
        let mut runtime = runtime();
        let (tx, rx) = mpsc::unbounded_channel();
        runtime.update_check_rx = Some(rx);
        tx.send(Ok(Some("0.5.3".into())))
            .expect("update check result should send");

        runtime.poll_update_check();

        let toast = runtime
            .state()
            .toast()
            .expect("update toast should be shown");
        assert_eq!(toast.kind, ToastKind::Info);
        assert!(toast.message.contains("0.5.3"));
        assert!(toast.message.contains("letcode update"));
        assert!(runtime.update_check_rx.is_none());
    }

    #[test]
    fn failed_update_check_is_silent() {
        let mut runtime = runtime();
        let (tx, rx) = mpsc::unbounded_channel();
        runtime.update_check_rx = Some(rx);
        tx.send(Err(anyhow::anyhow!("offline")))
            .expect("update check error should send");

        runtime.poll_update_check();

        assert!(runtime.state().toast().is_none());
        assert!(runtime.update_check_rx.is_none());
    }

    #[test]
    fn reads_named_and_detached_git_branches() {
        let path = temp_dir("repository");
        std::fs::create_dir_all(&path).expect("create temp repository");
        git(&path, &["init", "-q", "-b", "main"]);
        git(&path, &["config", "user.email", "test@example.com"]);
        git(&path, &["config", "user.name", "LetCode Test"]);
        std::fs::write(path.join("file.txt"), "content").expect("write file");
        git(&path, &["add", "file.txt"]);
        git(&path, &["commit", "-q", "-m", "initial"]);

        assert_eq!(read_git_branch(&path).as_deref(), Some("main"));
        git(&path, &["checkout", "-q", "--detach", "HEAD"]);
        let branch = read_git_branch(&path).expect("detached branch label");
        assert!(branch.starts_with("detached@"), "{branch}");

        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn non_git_directory_has_no_branch() {
        let path = temp_dir("non-repository");
        std::fs::create_dir_all(&path).expect("create temp directory");

        assert_eq!(read_git_branch(&path), None);

        let _ = std::fs::remove_dir_all(path);
    }
}

#[cfg(test)]
mod session_archive_tests {
    use super::TuiRuntime;
    use crate::session::archive::ArchiveRunReport;
    use crate::tui::TuiState;
    use crate::tui::state::ToastKind;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-session-archive-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = mpsc::unbounded_channel();
        TuiRuntime::new(
            TuiState::default(),
            rx,
            Vec::new(),
            Vec::new(),
            temp_dir("sessions"),
            temp_dir("preferences"),
        )
    }

    fn deliver_report(runtime: &mut TuiRuntime, report: ArchiveRunReport) {
        let (tx, rx) = mpsc::unbounded_channel();
        runtime.archive_pass_rx = Some(rx);
        tx.send(Ok(report)).expect("archive report should send");
    }

    #[test]
    fn archive_anomalies_surface_once_as_an_error_toast() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_language(Some(crate::tui::i18n::Language::En));
        deliver_report(
            &mut runtime,
            ArchiveRunReport {
                anomalies: vec!["broken".into()],
                ..ArchiveRunReport::default()
            },
        );

        runtime.poll_session_archive_pass();

        let toast = runtime.state().toast().expect("anomaly toast");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(
            toast.message,
            "Session archiving keeps failing for broken; see the log for details"
        );
        assert!(runtime.archive_pass_rx.is_none());
    }

    #[test]
    fn archive_anomalies_count_the_remaining_sessions() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_language(Some(crate::tui::i18n::Language::En));
        deliver_report(
            &mut runtime,
            ArchiveRunReport {
                anomalies: vec!["first".into(), "second".into(), "third".into()],
                ..ArchiveRunReport::default()
            },
        );

        runtime.poll_session_archive_pass();

        let toast = runtime.state().toast().expect("anomaly toast");
        assert_eq!(toast.kind, ToastKind::Error);
        assert_eq!(
            toast.message,
            "Session archiving keeps failing for first and 2 more sessions; see the log for details"
        );
    }

    #[test]
    fn a_successful_archive_pass_leaves_the_ui_unchanged() {
        let mut runtime = runtime();
        deliver_report(
            &mut runtime,
            ArchiveRunReport {
                archived_sessions: vec!["idle".into()],
                ..ArchiveRunReport::default()
            },
        );

        runtime.poll_session_archive_pass();

        assert!(runtime.state().toast().is_none());
        assert!(runtime.archive_pass_rx.is_none());
    }

    #[test]
    fn a_failed_archive_pass_stays_out_of_the_ui() {
        let mut runtime = runtime();
        let (tx, rx) = mpsc::unbounded_channel();
        runtime.archive_pass_rx = Some(rx);
        tx.send(Err(anyhow::anyhow!("unreadable sessions directory")))
            .expect("archive failure should send");

        runtime.poll_session_archive_pass();

        assert!(runtime.state().toast().is_none());
        assert!(runtime.archive_pass_rx.is_none());
    }
}

#[cfg(test)]
mod tests;
