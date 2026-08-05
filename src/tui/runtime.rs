use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::command::{
    ChildNavigation as SharedChildNavigation, CommandIntent, ThemeCommand, ToolOutputMode,
    TranscriptScrollbarMode, help_summary, parse_command,
};
use crate::mcp;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::skills::SkillCard;
#[cfg(test)]
use crate::transcript::SessionSummary;
use crate::transcript::{
    TranscriptEvent, TranscriptRecord, list_sessions, read_records, transcript_projection,
};
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
    ContextDetailTarget, DialogItem, DialogKind, DialogState, McpDiscoveryState,
    PendingQuestionState, QuestionAdvance, ToastKind, TranscriptClickTarget, TuiState,
};
use super::terminal::OwnedTerminal;
use super::theme::ThemeName;
#[cfg(test)]
use crate::session::RunnerPermissionRequest;
use crate::session::{RunnerQuestionRequest, SessionEngine, SessionTransportEvent};
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
use session_dialog::session_dialog_item;
#[cfg(test)]
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE_SCROLL_ROWS: u16 = 10;
const CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS: u8 = 20;
const TUI_FRAME_POLL_INTERVAL: Duration = Duration::from_millis(33);
const ASSISTANT_DELTA_BUFFER_MAX_BYTES: usize = 1024;
const ASSISTANT_DELTA_BUFFER_MAX_WAIT: Duration = Duration::from_millis(50);
const TERMINAL_TITLE_APP_NAME: &str = "LetCode";
const TERMINAL_TITLE_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TERMINAL_TITLE_TICKS_PER_FRAME: usize = 3;
const MCP_DISCOVERY_LOADING_DESCRIPTION: &str = "Discovering MCP servers";
const MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION: &str = "MCP discovery unavailable";
static NEXT_SUBMISSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ATTACHMENT_ID: AtomicU64 = AtomicU64::new(1);

fn mcp_discovery_description(discovery: McpDiscoveryState) -> Option<String> {
    match discovery {
        McpDiscoveryState::Loading => Some(MCP_DISCOVERY_LOADING_DESCRIPTION.into()),
        McpDiscoveryState::Ready => None,
        McpDiscoveryState::Unavailable => Some(MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION.into()),
    }
}

fn next_submission_id() -> String {
    format!(
        "user-submission-{}",
        NEXT_SUBMISSION_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn next_attachment_id() -> String {
    format!(
        "user-attachment-{}",
        NEXT_ATTACHMENT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn session_title_from_records(records: &[TranscriptRecord]) -> Option<String> {
    records.iter().rev().find_map(|record| match &record.event {
        TranscriptEvent::SessionTitle { title } => Some(title.clone()),
        _ => None,
    })
}

fn format_terminal_title(session_title: Option<&str>, spinner_frame: Option<usize>) -> String {
    let title = match session_title.filter(|title| !title.trim().is_empty()) {
        Some(title) => format!("{TERMINAL_TITLE_APP_NAME} | {title}"),
        None => TERMINAL_TITLE_APP_NAME.to_string(),
    };
    match spinner_frame {
        Some(frame) => format!(
            "{} {title}",
            TERMINAL_TITLE_SPINNER[frame % TERMINAL_TITLE_SPINNER.len()]
        ),
        None => title,
    }
}

fn assistant_delta_parts(
    event: &SessionTransportEvent,
) -> Option<(AssistantDeltaStream, Option<String>, String)> {
    match event {
        SessionTransportEvent::AssistantDelta(delta) => Some((
            AssistantDeltaStream {
                child_session_id: None,
                parent_tool_call_id: None,
                message_id: delta.message_id.clone(),
            },
            None,
            delta.delta.clone(),
        )),
        SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name,
            parent_tool_call_id,
            event: SessionEvent::AssistantDelta(delta),
        } => Some((
            AssistantDeltaStream {
                child_session_id: Some(child_session_id.clone()),
                parent_tool_call_id: parent_tool_call_id.clone(),
                message_id: delta.message_id.clone(),
            },
            agent_name.clone(),
            delta.delta.clone(),
        )),
        _ => None,
    }
}

fn assistant_delta_event(
    stream: &AssistantDeltaStream,
    agent_name: &Option<String>,
    delta: String,
) -> SessionTransportEvent {
    let delta = match &stream.message_id {
        Some(message_id) => {
            crate::tui::events::AssistantDeltaEvent::with_message_id(message_id, delta)
        }
        None => crate::tui::events::AssistantDeltaEvent::new(delta),
    };
    match &stream.child_session_id {
        Some(child_session_id) => SessionTransportEvent::ChildSessionEvent {
            child_session_id: child_session_id.clone(),
            agent_name: agent_name.clone(),
            parent_tool_call_id: stream.parent_tool_call_id.clone(),
            event: SessionEvent::AssistantDelta(delta),
        },
        None => SessionTransportEvent::AssistantDelta(delta),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AssistantDeltaStream {
    child_session_id: Option<String>,
    parent_tool_call_id: Option<String>,
    message_id: Option<String>,
}

#[derive(Debug)]
struct AssistantDeltaBuffer {
    stream: AssistantDeltaStream,
    agent_name: Option<String>,
    delta: String,
    started_at: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableModel {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub context_window_tokens: Option<u64>,
    pub reasoning_effort: Option<ModelReasoningEffort>,
    pub reasoning_efforts: Vec<ModelReasoningEffort>,
}

impl AvailableModel {
    #[cfg(test)]
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens: None,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn with_context_window(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
    ) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
        }
    }

    pub fn with_context_window_and_reasoning(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
        reasoning_effort: Option<ModelReasoningEffort>,
        reasoning_efforts: Vec<ModelReasoningEffort>,
    ) -> Self {
        let id = id.into();
        Self {
            provider: model_provider(&id),
            id,
            label: label.into(),
            context_window_tokens,
            reasoning_effort,
            reasoning_efforts,
        }
    }
}

fn model_provider(model_id: &str) -> String {
    model_id
        .split_once('/')
        .map(|(provider, _)| provider.to_string())
        .unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableExpert {
    pub agent_name: String,
    pub route_id: String,
}

/// Compatibility alias: session commands are owned by the backend boundary.
pub type RuntimeCommand = crate::session::SessionCommand;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupToast {
    message: String,
    kind: ToastKind,
}

impl StartupToast {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ToastKind::Success,
        }
    }

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

pub struct TuiRuntime {
    state: TuiState,
    session_transport_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
    permission_lifecycle: PermissionLifecycleController,
    pending_question_handle: Option<RunnerQuestionRequest>,
    pending_question_child_session_id: Option<String>,
    interrupt_confirmation_pending: bool,
    submitted_prompts: Vec<String>,
    queued_prompts: VecDeque<UserMessageSubmission>,
    queued_prompt_lifecycle: QueuedPromptLifecycle,
    session_turn_active: bool,
    session_resume_pending: bool,
    current_turn_output_tokens: u64,
    history_selection: Option<usize>,
    history_draft: Option<ComposerDraft>,
    available_models: Vec<AvailableModel>,
    available_experts: Vec<AvailableExpert>,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    assistant_delta_buffer: Option<AssistantDeltaBuffer>,
    session_title: Option<String>,
    spinner_frame: usize,
    theme_preview_original: Option<ThemeName>,
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
            queued_prompts: VecDeque::new(),
            queued_prompt_lifecycle: QueuedPromptLifecycle::default(),
            session_turn_active: false,
            session_resume_pending: false,
            current_turn_output_tokens: 0,
            history_selection: None,
            history_draft: None,
            available_models,
            available_experts,
            sessions_dir,
            preferences_dir,
            assistant_delta_buffer: None,
            session_title: None,
            spinner_frame: 0,
            theme_preview_original: None,
        }
    }

    pub fn state(&self) -> &TuiState {
        &self.state
    }

    #[cfg(test)]
    pub fn state_mut(&mut self) -> &mut TuiState {
        &mut self.state
    }

    fn terminal_title(&self) -> String {
        format_terminal_title(
            self.session_title.as_deref(),
            self.has_active_or_pending_session_turn()
                .then_some(self.spinner_frame / TERMINAL_TITLE_TICKS_PER_FRAME),
        )
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
        self.state.toast = None;
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
            self.state
                .show_toast("Answer every question before confirming", ToastKind::Info);
            return Ok(());
        }

        let Some(question) = self.state.pending_question.as_ref() else {
            self.state
                .show_toast("No question pending", ToastKind::Info);
            return Ok(());
        };
        if question.has_invalid_single_response() {
            self.state.show_toast(
                "Single-select questions accept only one answer",
                ToastKind::Info,
            );
            return Ok(());
        }

        let response = question.build_response();
        let handle = self.pending_question_handle.take();
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
        let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .filter(|question| question.editing_custom)
            .and_then(PendingQuestionState::current_question_mut)
        else {
            return;
        };
        question.custom_edit_cursor = question
            .custom_edit_cursor
            .min(question.custom_edit_text.len());
        question
            .custom_edit_text
            .insert_str(question.custom_edit_cursor, text);
        question.custom_edit_cursor += text.len();
    }

    fn backspace_pending_question_text(&mut self) {
        let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        else {
            return;
        };
        if question.custom_edit_cursor == 0 {
            return;
        }
        let previous = question.custom_edit_text[..question.custom_edit_cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        question
            .custom_edit_text
            .drain(previous..question.custom_edit_cursor);
        question.custom_edit_cursor = previous;
    }

    fn delete_pending_question_text(&mut self) {
        let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        else {
            return;
        };
        if question.custom_edit_cursor >= question.custom_edit_text.len() {
            return;
        }
        let next = question.custom_edit_text[question.custom_edit_cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| question.custom_edit_cursor + index)
            .unwrap_or(question.custom_edit_text.len());
        question
            .custom_edit_text
            .drain(question.custom_edit_cursor..next);
    }

    fn move_pending_question_cursor_left(&mut self) {
        let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        else {
            return;
        };
        if question.custom_edit_cursor == 0 {
            return;
        }
        question.custom_edit_cursor = question.custom_edit_text[..question.custom_edit_cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    fn move_pending_question_cursor_right(&mut self) {
        let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        else {
            return;
        };
        if question.custom_edit_cursor >= question.custom_edit_text.len() {
            return;
        }
        question.custom_edit_cursor = question.custom_edit_text[question.custom_edit_cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| question.custom_edit_cursor + index)
            .unwrap_or(question.custom_edit_text.len());
    }

    fn move_pending_question_cursor_home(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.custom_edit_cursor = 0;
        }
    }

    fn move_pending_question_cursor_end(&mut self) {
        if let Some(question) = self
            .state
            .pending_question
            .as_mut()
            .and_then(PendingQuestionState::current_question_mut)
        {
            question.custom_edit_cursor = question.custom_edit_text.len();
        }
    }

    pub fn try_drain_session_events(&mut self) {
        // Leave time in every frame for terminal input. In particular, an
        // unbounded stream of model deltas must not prevent a confirmed Esc
        // from reaching the session engine.
        const MAX_SESSION_EVENTS_PER_FRAME: usize = 256;
        self.flush_assistant_delta_buffer_if_due();
        for _ in 0..MAX_SESSION_EVENTS_PER_FRAME {
            match self.session_transport_rx.try_recv() {
                Ok(event) => self.consume_session_transport_event(event),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.handle_session_event_stream_closed();
                    break;
                }
            }
        }
    }

    fn handle_session_event_stream_closed(&mut self) {
        self.flush_assistant_delta_buffer();
        if self.has_active_or_pending_session_turn() || self.session_resume_pending {
            self.apply_session_transport_event(SessionTransportEvent::Error(ErrorEvent::new(
                "TUI session event stream closed unexpectedly",
            )));
            self.apply_session_transport_event(SessionTransportEvent::Done);
        }
    }

    fn consume_session_transport_event(&mut self, event: SessionTransportEvent) {
        if let Some((stream, agent_name, delta)) = assistant_delta_parts(&event) {
            self.buffer_assistant_delta(stream, agent_name, delta);
        } else {
            self.flush_assistant_delta_buffer();
            self.apply_session_transport_event(event);
        }
    }

    fn buffer_assistant_delta(
        &mut self,
        stream: AssistantDeltaStream,
        agent_name: Option<String>,
        delta: String,
    ) {
        if self
            .assistant_delta_buffer
            .as_ref()
            .is_some_and(|buffer| buffer.stream != stream)
        {
            self.flush_assistant_delta_buffer();
        }

        let buffer = self
            .assistant_delta_buffer
            .get_or_insert_with(|| AssistantDeltaBuffer {
                stream,
                agent_name,
                delta: String::new(),
                started_at: Instant::now(),
            });
        buffer.delta.push_str(&delta);

        if let Some(last_newline) = buffer.delta.rfind('\n') {
            let tail = buffer.delta.split_off(last_newline + 1);
            let committed = std::mem::replace(&mut buffer.delta, tail);
            let event = assistant_delta_event(&buffer.stream, &buffer.agent_name, committed);
            buffer.started_at = Instant::now();
            self.apply_session_transport_event(event);
        }

        if self
            .assistant_delta_buffer
            .as_ref()
            .is_some_and(|buffer| buffer.delta.len() >= ASSISTANT_DELTA_BUFFER_MAX_BYTES)
        {
            self.flush_assistant_delta_buffer();
        }
    }

    fn flush_assistant_delta_buffer_if_due(&mut self) {
        if self.assistant_delta_buffer.as_ref().is_some_and(|buffer| {
            !buffer.delta.is_empty()
                && buffer.started_at.elapsed() >= ASSISTANT_DELTA_BUFFER_MAX_WAIT
        }) {
            self.flush_assistant_delta_buffer();
        }
    }

    fn flush_assistant_delta_buffer(&mut self) {
        let Some(buffer) = self.assistant_delta_buffer.take() else {
            return;
        };
        if !buffer.delta.is_empty() {
            self.apply_session_transport_event(assistant_delta_event(
                &buffer.stream,
                &buffer.agent_name,
                buffer.delta,
            ));
        }
    }

    pub fn apply_session_transport_event(&mut self, event: SessionTransportEvent) {
        let mut suppress_session_event = false;

        match &event {
            SessionTransportEvent::QuestionRequested { request, handle } => {
                if self
                    .begin_pending_question(request.clone(), handle.clone(), None)
                    .is_err()
                {
                    let _ = handle.cancel("another interactive request is already pending");
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                    suppress_session_event = true;
                }
            }
            SessionTransportEvent::PermissionRequested { event, handle } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                    suppress_session_event = true;
                } else if let Err(handle) = self
                    .permission_lifecycle
                    .begin_parent(event.clone(), handle.clone())
                {
                    let _ = handle.deny();
                    self.state
                        .show_toast("Permission already pending", ToastKind::Info);
                    suppress_session_event = true;
                } else {
                    self.state.toast = None;
                }
            }
            SessionTransportEvent::ChildQuestionRequested {
                child_session_id,
                request,
                handle,
            } => {
                if self
                    .begin_pending_question(
                        request.clone(),
                        handle.clone(),
                        Some(child_session_id.clone()),
                    )
                    .is_err()
                {
                    let _ = handle.cancel("another interactive request is already pending");
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                    suppress_session_event = true;
                }
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
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                } else if let Err(handle) = self.permission_lifecycle.begin_child(
                    child_session_id.clone(),
                    event.clone(),
                    handle.clone(),
                ) {
                    let _ = handle.deny();
                    self.state
                        .show_toast("Permission already pending", ToastKind::Info);
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
            SessionTransportEvent::PermissionResolved(resolution) => {
                if self.pending_permission_matches_call(&resolution.call_id, None) {
                    self.permission_lifecycle.clear();
                }
            }
            SessionTransportEvent::Done => {
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
                self.interrupt_confirmation_pending = false;
                self.session_resume_pending = false;
                self.queued_prompt_lifecycle.record_error();
            }
            SessionTransportEvent::FastModeChanged { enabled } => {
                self.state.set_fast_mode_enabled(*enabled);
            }
            SessionTransportEvent::ModelChanged { model_id } => {
                self.apply_restored_model(model_id.clone());
                self.state.set_provider_label_from_model_route(model_id);
                self.show_toast("Model updated", ToastKind::Success);
            }
            SessionTransportEvent::ExpertModelChanged {
                agent_name,
                model_id,
            } => {
                if let Some(expert) = self
                    .available_experts
                    .iter_mut()
                    .find(|expert| expert.agent_name == *agent_name)
                {
                    expert.route_id = model_id.clone();
                }
                self.show_toast(format!("{agent_name} model updated"), ToastKind::Success);
            }
            SessionTransportEvent::QueuedPromptAccepted { prompt } => {
                self.queued_prompt_lifecycle.accept(&prompt.id);
            }
            SessionTransportEvent::SessionTitleUpdated { session_id, title } => {
                if self.state.session_id.as_deref() == Some(session_id) {
                    self.session_title = Some(title.clone());
                }
            }
            SessionTransportEvent::Interrupted => {
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
                if token_usage.output_tokens > 0 {
                    self.current_turn_output_tokens = self
                        .current_turn_output_tokens
                        .saturating_add(token_usage.output_tokens);
                }
                token_usage.output_tokens = self.current_turn_output_tokens;
                self.state
                    .apply_event(SessionEvent::TokenUsage(token_usage));
                suppress_session_event = true;
            }
            SessionTransportEvent::SessionTokenUsage(token_usage) => {
                // A committed manual compaction replaces the request snapshot.
                // Its local estimate has no provider response/cache accounting.
                self.current_turn_output_tokens = 0;
                self.state
                    .apply_event(SessionEvent::TokenUsage(token_usage.clone()));
                suppress_session_event = true;
            }
            SessionTransportEvent::ToolBatchFinished => {
                if !self.queued_prompts.is_empty()
                    && !self.queued_prompt_lifecycle.has_inflight_handoff()
                {
                    self.queued_prompt_lifecycle.mark_dispatch_ready();
                }
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
                        format!("Context projection failed: {error}"),
                        ToastKind::Error,
                    );
                    return;
                }
                self.session_resume_pending = false;
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
                self.state.set_current_context_branch(branch_id.clone());
                if let Some(token_usage) = token_usage {
                    self.state.set_token_usage(token_usage.clone().into());
                }
                self.state.show_toast("Session resumed", ToastKind::Info);
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
                if let Err(error) = self
                    .state
                    .try_replace_session_timeline_from_records_with_runtime_context(
                        records,
                        runtime_context.clone(),
                    )
                {
                    self.state.pending_question = pending_question;
                    self.state.show_toast(
                        format!("Context projection failed: {error}"),
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
                    self.apply_restored_model(model_id.clone());
                    self.state.set_provider_label_from_model_route(model_id);
                }
                self.state.set_current_context_branch(branch_id.clone());
                if let Some(token_usage) = token_usage {
                    self.state.set_token_usage(token_usage.clone().into());
                }
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
            } => {
                if self.state.child_view_metadata().is_some_and(|current| {
                    current.child_session_id == *child_session_id
                        && current.record_count == records.len()
                        && current.index == *index
                        && current.total == *total
                }) {
                    return;
                }
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
                    )
                {
                    self.state.show_toast(
                        format!("Context projection failed: {error}"),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state
                    .show_toast(format!("Viewing {agent_name}"), ToastKind::Info);
            }
            SessionTransportEvent::SessionHistoryLoaded { entries } => {
                self.open_history_tree_dialog(entries);
            }
            SessionTransportEvent::SessionStarted {
                session_id,
                records,
                runtime_context,
            } => {
                if let Err(error) = self
                    .state
                    .try_replace_session_timeline_from_records_with_runtime_context(
                        records,
                        runtime_context.clone(),
                    )
                {
                    self.state.show_toast(
                        format!("Context projection failed: {error}"),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state.session_id = Some(session_id.clone());
                self.session_title = session_title_from_records(records);
                self.permission_lifecycle.clear();
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.session_turn_active = false;
                self.current_turn_output_tokens = 0;
                self.state.timeline.remove_queued_user_message_previews();
                // A newly created, still-empty session remains on the dashboard.
                self.state.active_session = false;
                self.state
                    .set_current_context_branch(crate::transcript::ROOT_CONTEXT_BRANCH_ID);
                self.state
                    .show_toast("New session started", ToastKind::Info);
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
                if self.child_event_clears_pending_permission(child_session_id, event) {
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
                if matches!(
                    event,
                    SessionEvent::Error(_) | SessionEvent::Done | SessionEvent::Interrupted
                ) {
                    self.interrupt_confirmation_pending = false;
                }
                self.state.apply_child_session_event_with_agent(
                    child_session_id,
                    agent_name.as_deref(),
                    parent_tool_call_id.as_deref(),
                    event.clone(),
                );
            }
            _ => {}
        }

        self.reproject_pending_permission();

        if !suppress_session_event {
            if let Some(session_event) = event.session_event() {
                self.state.apply_event(session_event);
                self.reproject_pending_permission();
            }
        }
    }

    fn reproject_pending_permission(&mut self) {
        self.state
            .set_pending_permission_projection(self.permission_lifecycle.projection());
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
            InputAction::Interrupt | InputAction::Tick | InputAction::ChildPrefix
        ) {
            self.interrupt_confirmation_pending = false;
        }

        if !matches!(
            action,
            InputAction::NoOp | InputAction::Tick | InputAction::ChildPrefix
        ) {
            self.state.child_navigation_prefix = false;
        }

        if apply_edit_action(&mut self.state, &action) {
            if matches!(
                action,
                InputAction::Insert(_)
                    | InputAction::Paste(_)
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
            InputAction::CycleReasoningEffort => {
                if self.state.is_read_only_child_view() {
                    self.push_child_view_read_only_notice();
                    Ok(None)
                } else {
                    Ok(self.cycle_reasoning_effort_command())
                }
            }
            InputAction::ChildPrefix => {
                self.state.child_navigation_prefix = true;
                self.state.child_navigation_prefix_ticks_remaining =
                    CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS;
                self.state.show_toast("Child navigation", ToastKind::Info);
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
            InputAction::ChildParent => {
                if self.state.is_read_only_child_view() {
                    self.state.restore_parent_timeline_view();
                    self.state.show_toast("Parent transcript", ToastKind::Info);
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
                        match question.pick_row(question.active_row) {
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
                    if dialog.kind == DialogKind::ContextPicker && dialog.detail_focused {
                        dialog.scroll_detail_next();
                    } else {
                        dialog.select_next();
                    }
                }
                self.sync_context_inspector_preview();
                self.preview_selected_theme();
                Ok(None)
            }
            InputAction::DialogPrev => {
                if let Some(dialog) = self.state.dialog_mut() {
                    if dialog.kind == DialogKind::ContextPicker && dialog.detail_focused {
                        dialog.scroll_detail_previous();
                    } else {
                        dialog.select_previous();
                    }
                }
                self.sync_context_inspector_preview();
                self.preview_selected_theme();
                Ok(None)
            }
            InputAction::DialogAccept => self.handle_dialog_accept(),
            InputAction::DialogToggle => self.handle_mcp_toggle(),
            InputAction::DialogCancel => {
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
                    self.state.show_toast("Context", ToastKind::Info);
                } else {
                    self.state.close_dialog();
                    self.state.show_toast("Dialog closed", ToastKind::Info);
                }
                Ok(None)
            }
            InputAction::DialogInsert(ch) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.insert_query_char(ch);
                }
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::DialogPaste(text) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    for ch in text.chars() {
                        dialog.insert_query_char(ch);
                    }
                }
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::DialogBackspace => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.pop_query_char();
                }
                self.state.sync_context_picker_preview();
                Ok(None)
            }
            InputAction::Submit => self.handle_submit(),
            InputAction::HistoryPrev => {
                self.navigate_history_previous();
                Ok(None)
            }
            InputAction::HistoryNext => {
                self.navigate_history_next();
                Ok(None)
            }
            InputAction::ApprovePermission => {
                if let Some(handle) = self.permission_lifecycle.take_handle() {
                    handle.approve()?;
                }
                Ok(None)
            }
            InputAction::ApprovePermissionAlways => {
                if self
                    .state
                    .pending_permission
                    .as_ref()
                    .is_some_and(|permission| permission.can_allow_always)
                {
                    if let Some(handle) = self.permission_lifecycle.take_handle() {
                        handle.allow_always()?;
                    }
                }
                Ok(None)
            }
            InputAction::DenyPermission => {
                if let Some(handle) = self.permission_lifecycle.take_handle() {
                    handle.deny()?;
                }
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
            InputAction::MouseSelectionEnd(col, row) => {
                self.handle_selection_end(col, row);
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
                        self.state.show_toast("Ready", ToastKind::Info);
                    }
                }
                self.state.apply_event(SessionEvent::Tick);
                self.spinner_frame = self.spinner_frame.wrapping_add(1);
                self.tick_selection_autoscroll();
                Ok(None)
            }
            InputAction::Insert(_)
            | InputAction::Paste(_)
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
        drawer.draw(&mut self.state)
    }

    fn has_active_or_pending_session_turn(&self) -> bool {
        has_active_or_pending_session_turn(active_turn_state(
            &self.state,
            self.session_turn_active,
            self.queued_prompt_lifecycle.has_inflight_handoff(),
            self.permission_lifecycle.is_pending(),
        ))
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
                        .show_toast("Remove attachments before running command", ToastKind::Info);
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
                .show_toast("Remove attachments before running command", ToastKind::Info);
            return Ok(None);
        }
        self.reset_history_navigation();
        let active_session_turn = self.has_active_or_pending_session_turn();
        let active_turn_command_allowed = matches!(
            &parsed_command,
            Ok(CommandIntent::Help
                | CommandIntent::ContextBrowse
                | CommandIntent::McpBrowse
                | CommandIntent::SkillBrowse
                | CommandIntent::ToolOutputSet(_)
                | CommandIntent::TranscriptScrollbarSet(_)
                | CommandIntent::Theme(_)
                | CommandIntent::Child(_)
                | CommandIntent::Parent)
        );
        if active_session_turn && !active_turn_command_allowed {
            if matches!(&parsed_command, Ok(CommandIntent::Delegate { .. })) {
                self.state.show_toast("Turn still running", ToastKind::Info);
                return Ok(None);
            }

            if matches!(&parsed_command, Ok(CommandIntent::Prompt(_)))
                && !self.state.is_read_only_child_view()
            {
                self.queue_prompt(UserMessageSubmission::new(next_submission_id(), content));
                return Ok(None);
            }

            self.state.show_toast("Turn still running", ToastKind::Info);
            return Ok(None);
        }

        if self.state.is_read_only_child_view() && !child_view_allows_prompt(&command_input) {
            self.push_child_view_read_only_notice();
            return Ok(None);
        }

        if let Some(command) = self.handle_parsed_command(parsed_command)? {
            self.state.clear_input();
            return Ok(match command {
                SubmittedCommand::LocalOnly => None,
                SubmittedCommand::Runtime(command) => Some(command),
            });
        }

        self.state.clear_input();
        self.state.clear_composer_tokens();
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        self.queued_prompt_lifecycle.clear_dispatch_ready();
        self.session_turn_active = true;
        self.state.toast = None;
        self.submitted_prompts.push(prompt.clone());

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
        self.state
            .set_input(self.submitted_prompts[next_index].clone());
    }

    fn navigate_history_next(&mut self) {
        let Some(index) = self.history_selection else {
            return;
        };

        if index + 1 < self.submitted_prompts.len() {
            let next_index = index + 1;
            self.history_selection = Some(next_index);
            self.state
                .set_input(self.submitted_prompts[next_index].clone());
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

    fn reset_history_navigation(&mut self) {
        self.history_selection = None;
        self.history_draft = None;
    }

    fn queue_prompt(&mut self, prompt: UserMessageSubmission) {
        self.state.clear_input();
        self.state.clear_composer_tokens();
        self.state.mark_session_active();
        self.submitted_prompts.push(prompt.content.text.clone());
        self.queued_prompts.push_back(prompt.clone());
        self.state.push_queued_user_message_preview(prompt);
        self.state.toast = None;
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
            self.state
                .show_toast("Press Esc again to interrupt", ToastKind::Info);
            return Ok(None);
        }

        self.interrupt_confirmation_pending = false;
        self.state.show_toast("Interrupting", ToastKind::Info);
        Ok(Some(RuntimeCommand::Interrupt))
    }

    fn handle_parsed_command(
        &mut self,
        parsed: Result<CommandIntent, crate::command::CommandParseError>,
    ) -> Result<Option<SubmittedCommand>> {
        let intent = match parsed {
            Ok(intent) => intent,
            Err(error) => {
                self.push_command_notice(error.message());
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
            CommandIntent::Prompt(_) => Ok(None),
            CommandIntent::Exit => {
                self.state.apply_event(SessionEvent::Quit);
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::Help => {
                self.push_command_notice(help_summary());
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::ModelShow => self.show_model_dialog(),
            CommandIntent::AgentsShow => self.show_agents_dialog(),
            CommandIntent::ReasoningShow => self.show_reasoning_dialog(),
            CommandIntent::PermissionShow => self.show_permission_dialog(),
            CommandIntent::ToolOutputSet(mode) => self.handle_tool_output_command(mode),
            CommandIntent::Theme(command) => Ok(Some(self.handle_theme_command(command))),
            CommandIntent::TranscriptScrollbarSet(mode) => {
                Ok(Some(self.handle_transcript_scrollbar_command(mode)))
            }
            CommandIntent::ResumeShow => self.show_resume_dialog(),
            CommandIntent::ContextBrowse => self.show_context_dialog(),
            CommandIntent::McpBrowse => self.show_mcp_dialog(),
            CommandIntent::SkillBrowse => self.show_skill_dialog(),
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
            | CommandIntent::Child(_)
            | CommandIntent::Parent => unreachable!(
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
            SessionCommand::SetExpertModel {
                agent_name,
                model_id,
            } => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::SetExpertModel {
                    agent_name,
                    model_id,
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
            SessionCommand::Compact => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.session_turn_active = true;
                self.state.show_toast("Compacting context", ToastKind::Info);
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Compact)))
            }
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
                self.state.show_toast("Resuming session", ToastKind::Info);
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
            } => Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewChild {
                navigation,
                anchor_child_session_id,
            }))),
            SessionCommand::ViewParent => {
                if self.state.transcript_view.is_child() {
                    self.state.restore_parent_timeline_view();
                    self.state.show_toast("Parent transcript", ToastKind::Info);
                }
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewParent)))
            }
            SessionCommand::DelegateSubagent { agent_name, task } => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.session_turn_active = true;
                self.state
                    .timeline
                    .push_delegation(agent_name.clone(), task.clone());
                self.state.toast = None;
                Ok(Some(SubmittedCommand::Runtime(
                    RuntimeCommand::DelegateSubagent { agent_name, task },
                )))
            }
            SessionCommand::ToggleMcpServer(server_name) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ToggleMcpServer(server_name),
            ))),
            SessionCommand::Interrupt => {
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Interrupt)))
            }
        }
    }

    fn handle_tool_output_command(
        &mut self,
        mode: ToolOutputMode,
    ) -> Result<Option<SubmittedCommand>> {
        let mode = match mode {
            ToolOutputMode::Toggle => {
                if self.state.tool_output_expanded {
                    LocalToolOutputMode::Truncated
                } else {
                    LocalToolOutputMode::Expanded
                }
            }
            ToolOutputMode::Expanded => LocalToolOutputMode::Expanded,
            ToolOutputMode::Truncated => LocalToolOutputMode::Truncated,
        };

        self.state.set_tool_output_expanded(mode.expanded());
        let prefs = self.tui_preferences();
        if let Err(_error) = prefs.save_to_dir(&self.preferences_dir) {
            self.state
                .show_toast("Tool output mode changed", ToastKind::Info);
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        self.state
            .show_toast("Tool output mode changed", ToastKind::Info);
        Ok(Some(SubmittedCommand::LocalOnly))
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
        let prefs = self.tui_preferences();
        if let Err(_error) = prefs.save_to_dir(&self.preferences_dir) {
            self.state
                .show_toast("Transcript scrollbar", ToastKind::Info);
            return SubmittedCommand::LocalOnly;
        }
        self.state
            .show_toast("Transcript scrollbar", ToastKind::Info);
        SubmittedCommand::LocalOnly
    }

    fn tui_preferences(&self) -> TuiPreferences {
        TuiPreferences {
            tool_output_expanded: self.state.tool_output_expanded,
            transcript_scrollbar_visible: self.state.transcript_scrollbar_visible,
            theme: self.state.theme_name,
        }
    }

    fn handle_theme_command(&mut self, command: ThemeCommand) -> SubmittedCommand {
        match command {
            ThemeCommand::Show => self.show_theme_dialog(),
            ThemeCommand::Set(theme) => self.apply_theme_selection(theme),
        }
        SubmittedCommand::LocalOnly
    }

    fn show_theme_dialog(&mut self) {
        self.theme_preview_original = Some(self.state.theme_name);
        let items = vec![
            DialogItem::new("dark", "Dark", Some("Neutral dark palette".into())),
            DialogItem::new("ocean", "Ocean", Some("Cool blue and teal palette".into())),
            DialogItem::new("forest", "Forest", Some("Natural green palette".into())),
            DialogItem::new("rose", "Rose", Some("Warm rose palette".into())),
            DialogItem::new("rainbow", "Rainbow", Some("Animated accent colors".into())),
        ];
        let mut dialog = DialogState::new(
            DialogKind::ThemePicker,
            "Select theme",
            Some("Choose the TUI color palette".into()),
            items,
        );
        dialog.selected = ThemeName::available()
            .iter()
            .position(|theme| *theme == self.state.theme_name)
            .unwrap_or_default();
        self.state.open_dialog(dialog);
    }

    fn apply_theme_selection(&mut self, theme: ThemeName) {
        self.theme_preview_original = None;
        self.state.set_theme_name(theme);
        let prefs = self.tui_preferences();
        if let Err(error) = prefs.save_to_dir(&self.preferences_dir) {
            tracing::warn!(%error, "failed to save TUI preferences");
            self.state
                .show_toast("Theme changed; preference not saved", ToastKind::Info);
        } else {
            self.state
                .show_toast(format!("Theme: {}", theme.as_str()), ToastKind::Info);
        }
    }

    fn preview_selected_theme(&mut self) {
        let theme = self.state.dialog().and_then(|dialog| {
            (dialog.kind == DialogKind::ThemePicker)
                .then(|| dialog.selected_item())
                .flatten()
                .and_then(|item| ThemeName::parse(&item.id))
        });
        if let Some(theme) = theme {
            self.state.set_theme_name(theme);
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
        if let Some(theme) = self.theme_preview_original.take() {
            self.state.set_theme_name(theme);
        }
        self.state.close_dialog();
        self.state.show_toast("Dialog closed", ToastKind::Info);
        true
    }

    fn show_permission_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = vec![
            DialogItem::new("safe", "Safe", Some("Ask before all tools".into())),
            DialogItem::new(
                "default",
                "Default",
                Some("Allow read/preview, ask for risky tools".into()),
            ),
            DialogItem::new(
                "yolo",
                "YOLO",
                Some("Allow write and command tools without asking".into()),
            ),
        ];
        let mut dialog = DialogState::new(
            DialogKind::PermissionPicker,
            "Permission mode",
            Some("Select how much freedom the agent has when using tools".into()),
            items,
        );
        dialog.selected = match self.state.permission_mode_label.as_str() {
            "safe" => 0,
            "yolo" => 2,
            _ => 1,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
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
        let mut dialog = DialogState::new(DialogKind::ModelPicker, "Select model", None, items);
        if let Some(index) = self
            .available_models
            .iter()
            .position(|model| model.id == self.state.model_id)
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

    fn show_agents_dialog_with_state(&mut self, query: String, selected_agent: Option<String>) {
        let items = self
            .available_experts
            .iter()
            .map(|expert| {
                DialogItem::new(expert.agent_name.clone(), expert.agent_name.clone(), None)
                    .with_section("Experts")
                    .with_right_detail(expert.route_id.clone())
            })
            .collect();
        let mut dialog = DialogState::new(DialogKind::AgentPicker, "Expert models", None, items);
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
        let current_route = self
            .available_experts
            .iter()
            .find(|expert| expert.agent_name == agent_name)
            .map(|expert| expert.route_id.clone())
            .unwrap_or_else(|| self.state.model_id.clone());
        let mut dialog = DialogState::new(
            DialogKind::ExpertModelPicker(agent_name),
            "Select expert model",
            None,
            self.model_dialog_items(),
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
            self.push_command_notice(format!(
                "Unknown model: {}. Available models: {available}",
                model_id
            ));
            return Ok(Some(SubmittedCommand::LocalOnly));
        };

        Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::SetModel(
            model.id,
        ))))
    }

    fn show_resume_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let sessions = list_sessions(&self.sessions_dir)?;
        if sessions.is_empty() {
            self.push_command_notice("No previous sessions found");
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        let items = sessions.iter().map(session_dialog_item).collect::<Vec<_>>();
        let dialog = DialogState::new(DialogKind::SessionPicker, "Sessions", None, items);
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn open_history_tree_dialog(&mut self, entries: &[transcript_projection::SessionHistoryEntry]) {
        if entries.is_empty() {
            self.push_command_notice("No transcript entries found");
            return;
        }

        let mut dialog = DialogState::new(
            DialogKind::HistoryTree,
            "Session history",
            Some("Select an entry".into()),
            history_tree_dialog_items(entries),
        );
        dialog.selected = dialog.items.len().saturating_sub(1);
        self.state.open_dialog(dialog);
    }

    fn show_reasoning_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let efforts = self.active_reasoning_efforts();
        if efforts.is_empty() {
            self.push_command_notice("The selected model does not support configurable reasoning");
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            "Reasoning effort",
            Some("Select how much reasoning the model should use".into()),
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
            self.push_command_notice("No context details found");
            return Ok(Some(SubmittedCommand::LocalOnly));
        }
        let mut dialog = DialogState::new(DialogKind::ContextPicker, "Context", None, items);
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
            "MCP Servers",
            description,
            mcp_dialog_items(&self.state.mcp_servers, &self.state.mcp_updating),
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
        let items = mcp_dialog_items(&self.state.mcp_servers, &self.state.mcp_updating);
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
            "Local Skills",
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
            format!("MCP Tools · {server_name}"),
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
                mcp::McpServerStatus::Disabled => "Disabled · cached tools are not callable".into(),
                mcp::McpServerStatus::Online { tool_count } => {
                    format!("Online · {tool_count} tools available")
                }
                mcp::McpServerStatus::Offline { .. } => "Offline".into(),
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
        if self.session_turn_active {
            self.show_toast(
                "MCP changes unavailable while a turn is active",
                ToastKind::Error,
            );
            return Ok(None);
        }
        if self.state.mcp_updating.contains(&server_name) {
            self.show_toast("MCP server update is still in progress", ToastKind::Error);
            return Ok(None);
        }
        self.state
            .set_mcp_server_updating(server_name.clone(), true);
        self.refresh_open_mcp_dialog();
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
                Ok(Some(RuntimeCommand::SetModel(selected.id)))
            }
            DialogKind::AgentPicker => {
                self.show_expert_model_dialog(selected.id);
                Ok(None)
            }
            DialogKind::ExpertModelPicker(agent_name) => {
                self.state.close_dialog();
                Ok(Some(RuntimeCommand::SetExpertModel {
                    agent_name,
                    model_id: selected.id,
                }))
            }
            DialogKind::PermissionPicker => {
                self.state.close_dialog();
                let mode = match selected.id.as_str() {
                    "safe" => PermissionMode::Safe,
                    "yolo" => PermissionMode::Yolo,
                    _ => PermissionMode::Default,
                };
                let label = mode.to_string();
                self.state.set_permission_mode_label(label.clone());
                self.show_toast(
                    format!("Permission mode updated · {label}"),
                    ToastKind::Success,
                );
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
                self.state
                    .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(
                        effort.clone(),
                    ))));
                Ok(Some(RuntimeCommand::SetReasoningEffort(effort)))
            }
            DialogKind::ThemePicker => {
                self.state.close_dialog();
                let theme = ThemeName::parse(&selected.id)
                    .expect("theme picker items should use valid theme ids");
                self.apply_theme_selection(theme);
                Ok(None)
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
                if !detail_focused && self.state.active_context_open_detail().is_some() {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.detail_focused = true;
                        dialog.detail_scroll = 0;
                    }
                }
                Ok(None)
            }
            DialogKind::SkillPicker => {
                let attached = self.state.add_composer_skill(selected.id);
                self.state.close_dialog();
                self.show_toast(
                    if attached {
                        "Skill attached"
                    } else {
                        "Skill already attached"
                    },
                    ToastKind::Success,
                );
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
                "Context item unavailable",
                "Refresh context and try again",
            );
            return;
        };

        if !context_detail_available(self.state.active_context(), &target) {
            self.state.open_context_detail(None);
            self.notify_context_dialog_issue(
                "Context item unavailable",
                "Refresh context and try again",
            );
            return;
        }

        self.state.open_context_detail(Some(target));
    }

    fn set_permission_mode_command(&mut self, mode: PermissionMode) -> SubmittedCommand {
        let label = mode.to_string();
        self.state.set_permission_mode_label(label.clone());
        self.show_toast(
            format!("Permission mode updated · {label}"),
            ToastKind::Success,
        );
        SubmittedCommand::Runtime(RuntimeCommand::SetPermissionMode(mode))
    }

    fn set_reasoning_effort_command(&mut self, effort: ModelReasoningEffort) -> SubmittedCommand {
        if !self.active_reasoning_efforts().contains(&effort) {
            self.push_command_notice(
                "That reasoning effort is not supported by the selected model",
            );
            return SubmittedCommand::LocalOnly;
        }
        self.state
            .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(effort.clone()))));
        SubmittedCommand::Runtime(RuntimeCommand::SetReasoningEffort(effort))
    }

    fn cycle_reasoning_effort_command(&mut self) -> Option<RuntimeCommand> {
        let efforts = self.active_reasoning_efforts();
        let Some(next) = next_reasoning_effort(&efforts, self.current_reasoning_effort()) else {
            self.push_command_notice("The selected model does not support configurable reasoning");
            return None;
        };
        self.state
            .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(next.clone()))));
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
                .reasoning_effort_label
                .as_deref()
                .unwrap_or("off"),
        ) {
            Some(ModelReasoningEffort::None) | None => None,
            Some(effort) => Some(effort),
        }
    }

    fn push_command_notice(&mut self, message: impl Into<String>) {
        self.state.show_toast(message.into(), ToastKind::Info);
    }

    fn push_child_view_read_only_notice(&mut self) {
        self.state
            .show_toast("Viewing child transcript", ToastKind::Info);
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

    fn handle_transcript_click(&mut self, col: u16, row: u16) {
        if let Some(TranscriptClickTarget::ToolCard(call_id)) =
            self.state.transcript_click_target(col, row)
        {
            self.state.toggle_tool_output(&call_id);
        }
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
        self.state.selection_dragged = true;
        self.state.selection_last_mouse = Some((col, row));
        if let Some(anchor) = self.state.map_mouse_to_anchor(col, row) {
            if let Some(selection) = &mut self.state.text_selection {
                selection.end = anchor;
            }
        }
    }

    fn handle_selection_end(&mut self, col: u16, row: u16) {
        let dragged = self.state.selection_dragged;
        if dragged {
            self.handle_selection_drag(col, row);
        }
        self.state.selection_in_progress = false;
        self.state.selection_dragged = false;
        self.state.selection_last_mouse = None;
        // 抛弃零宽选择（单击未拖动），避免接管 Ctrl+C 复制语义且无视觉反馈
        if let Some(selection) = &self.state.text_selection {
            if selection.start == selection.end {
                self.state.text_selection = None;
            }
        }
        if !dragged {
            self.handle_transcript_click(col, row);
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
        if let Some(anchor) = self.state.map_mouse_to_anchor(clamped_col, clamped_row) {
            if let Some(selection) = &mut self.state.text_selection {
                selection.end = anchor;
            }
        }
    }

    fn handle_copy_selection(&mut self) -> Result<()> {
        use arboard::Clipboard;

        let text = crate::tui::selection::extract_selected_text(&self.state);
        if text.is_empty() {
            return Ok(());
        }

        match Clipboard::new() {
            Ok(mut clipboard) => {
                if clipboard.set_text(text).is_err() {
                    self.show_toast("Couldn’t copy to clipboard", ToastKind::Error);
                } else {
                    self.show_toast("Copied to clipboard", ToastKind::Success);
                }
            }
            Err(_) => {
                self.show_toast("Clipboard unavailable", ToastKind::Error);
            }
        }

        Ok(())
    }

    fn handle_paste_from_clipboard(&mut self) -> Result<()> {
        use arboard::Clipboard;
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

        match Clipboard::new() {
            Ok(mut clipboard) => {
                if let Ok(text) = clipboard.get_text()
                    && !text.is_empty()
                {
                    let action = map_paste_event(&self.state, text);
                    let _ = self.handle_input_action(action)?;
                    return Ok(());
                }

                // Follow the opencode-style fallback order: if there is no plain text,
                // try to read a native clipboard image and attach it to the composer draft.
                if !self.state.dialog_is_open() && self.state.pending_permission.is_none() {
                    if let Ok(image) = clipboard.get_image() {
                        let mut png_bytes = Vec::new();
                        PngEncoder::new(&mut png_bytes).write_image(
                            image.bytes.as_ref(),
                            image.width as u32,
                            image.height as u32,
                            ColorType::Rgba8.into(),
                        )?;

                        let data_url =
                            format!("data:image/png;base64,{}", STANDARD.encode(png_bytes));
                        self.state.add_composer_attachment(UserImageAttachment {
                            id: next_attachment_id(),
                            label: "clipboard".into(),
                            mime: "image/png".into(),
                            data_url,
                        });
                        self.reset_history_navigation();
                        self.show_toast("Image added", ToastKind::Success);
                        return Ok(());
                    }
                }

                self.show_toast("Couldn’t paste from clipboard", ToastKind::Error);
            }
            Err(_) => {
                self.show_toast("Clipboard unavailable", ToastKind::Error);
            }
        }

        Ok(())
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
        .cloned()
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
                reasoning_effort_config_label(&effort),
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
            | "/children"
            | "/parent"
            | "/tool-output"
            | "/scrollbar"
            | "/theme"
            | "/context"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalToolOutputMode {
    Expanded,
    Truncated,
}

impl LocalToolOutputMode {
    fn expanded(self) -> bool {
        matches!(self, Self::Expanded)
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

pub async fn run_tui(
    mut engine: SessionEngine,
    projection: crate::session::SessionEngineProjection,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    provider_label: String,
    available_models: Vec<AvailableModel>,
    available_experts: Vec<AvailableExpert>,
    startup_toast: Option<StartupToast>,
    skill_cards: Vec<SkillCard>,
) -> Result<()> {
    let mut state = TuiState::new(
        projection.model_id,
        projection.model_label,
        projection.permission_mode_label,
    );
    state.session_id = Some(projection.session_id);
    state.set_skill_cards(skill_cards);
    let preferences = TuiPreferences::load_from_dir(&preferences_dir);
    state.set_tool_output_expanded(preferences.tool_output_expanded);
    state.set_transcript_scrollbar_visible(preferences.transcript_scrollbar_visible);
    state.set_theme_name(preferences.theme);
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
        state.show_toast("Missing API key", ToastKind::Info);
    }
    if let Some(toast) = startup_toast {
        state.show_toast(toast.message, toast.kind);
    }

    let ingress = engine.take_ingress();
    let session_transport_rx = engine.take_event_egress().into_receiver();
    let tui_result = async {
        let mut runtime = TuiRuntime::new(
            state,
            session_transport_rx,
            available_models,
            available_experts,
            sessions_dir,
            preferences_dir,
        );
        runtime.session_title = projection.session_title;
        let mut terminal = OwnedTerminal::new()?;
        runtime.update_terminal_title(&mut terminal)?;
        let mut drawer = TerminalDrawer::new(&mut terminal);

        loop {
            runtime.try_drain_session_events();
            if let Some(command) = runtime.take_next_queued_prompt_command() {
                command_dispatch::dispatch_command(&mut runtime, command, &ingress, true);
            }
            drawer.set_title(&runtime.terminal_title())?;
            runtime.draw(&mut drawer)?;

            if runtime.state().quit_requested {
                break;
            }
            if event::poll(TUI_FRAME_POLL_INTERVAL)? {
                match event::read()? {
                    Event::Key(key) => {
                        let action = map_key_event(runtime.state(), key);
                        if let Some(command) = runtime.handle_input_action(action)? {
                            command_dispatch::dispatch_command(
                                &mut runtime,
                                command,
                                &ingress,
                                true,
                            );
                        }
                    }
                    Event::Mouse(mouse) => {
                        let action = map_mouse_event(runtime.state(), mouse);
                        if let Some(command) = runtime.handle_input_action(action)? {
                            command_dispatch::dispatch_command(
                                &mut runtime,
                                command,
                                &ingress,
                                false,
                            );
                        }
                    }
                    Event::Paste(text) => {
                        let action = map_paste_event(runtime.state(), text);
                        if let Some(command) = runtime.handle_input_action(action)? {
                            command_dispatch::dispatch_command(
                                &mut runtime,
                                command,
                                &ingress,
                                false,
                            );
                        }
                    }
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            } else {
                let _ = runtime.handle_input_action(InputAction::Tick)?;
            }
        }
        Ok(())
    }
    .await;

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
        // Keep the hardware cursor hidden for the whole frame flush. Ratatui only
        // re-hides after flush when no frame cursor is set; if anything leaves the
        // cursor visible, full-screen redraws make it appear to jump around.
        let terminal = self.terminal.terminal_mut();
        // Hide before and after paint: ratatui only re-hides after flush when the
        // frame does not set a cursor position, so a briefly-visible caret during
        // buffer writes looks like it is jumping across the UI.
        let _ = terminal.hide_cursor();
        let completed = terminal.draw(|frame| render::render(frame, state))?;
        let overlay = super::transcript_ratatui::plan_hyperlink_overlay(
            completed.buffer,
            &self.applied_hyperlink_cells,
            &state.frame_hyperlink_cells,
        );
        super::transcript_ratatui::write_hyperlink_overlay(terminal.backend_mut(), &overlay)?;
        self.applied_hyperlink_cells = overlay.applied;
        let _ = terminal.hide_cursor();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
