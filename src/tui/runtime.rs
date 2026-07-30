use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent, ManualCompactionOutcome, SubagentInvocation};
use crate::agent_event_journal::persist_agent_event;
use crate::command::{
    ChildNavigation as SharedChildNavigation, CommandIntent, ToolOutputMode,
    TranscriptScrollbarMode, help_summary, parse_command,
};
use crate::mcp;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::runtime_context::RuntimeActiveContext;
use crate::skills::SkillCard;
use crate::subagent::SubagentPool;
use crate::tool::{ToolHandler, normalize_subagent_input};
#[cfg(test)]
use crate::transcript::SessionSummary;
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord, TranscriptRecorder, list_sessions,
    read_child_session_records_allow_partial_tail, read_records, remove_empty_session_file,
    sync_recorder_branch, transcript_projection,
};
use crate::user_content::{UserImageAttachment, UserMessageSubmission};

use super::catalog::{mcp_dialog_items, mcp_tool_dialog_items, skill_dialog_items};
use super::events::{
    AppEvent, ErrorEvent, NoticeEvent, RuntimeContextDisposition, RuntimeContextUpdatedEvent,
    TokenUsageEvent,
};
use super::input::{
    InputAction, apply_edit_action, map_key_event, map_mouse_event, map_paste_event,
};
use super::preferences::TuiPreferences;
use super::render;
use super::slash::{SlashCommandEntry, matching_completion_commands};
use super::state::{
    ContextDetailTarget, DialogItem, DialogKind, DialogState, McpDiscoveryState,
    PendingQuestionState, QuestionAdvance, ToastKind, TuiState,
};
use super::terminal::OwnedTerminal;
use crate::session::{AgentRunner, RunnerEvent, RunnerPermissionRequest, RunnerQuestionRequest};
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
#[path = "runtime/restore_projection.rs"]
mod restore_projection;
#[path = "runtime/session_cleanup.rs"]
mod session_cleanup;
#[path = "runtime/session_command_adapter.rs"]
mod session_command_adapter;
#[path = "runtime/session_dialog.rs"]
mod session_dialog;
use crate::session::{restored_session_token_usage, session_resumed_event, session_started_event};
use async_openai::config::Config;
use history_tree_dialog::history_tree_dialog_items;
use lifecycle::{active_turn_state, build_interrupt_request, has_active_or_pending_runner_turn};
use permission_lifecycle::PermissionLifecycleController;
use queued_prompt::{QueuedPromptDoneDisposition, QueuedPromptLifecycle};
use restore_projection::{
    project_runtime_restore_snapshot_with_children, runtime_context_from_records,
};
use serde_json::json;
use session_cleanup::{empty_session_path, remove_current_empty_session};
use session_dialog::session_dialog_item;
use std::sync::atomic::AtomicU64;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, Ordering},
};

const PAGE_SCROLL_ROWS: u16 = 10;
const CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS: u8 = 20;
const TUI_FRAME_POLL_INTERVAL: Duration = Duration::from_millis(33);
const ASSISTANT_DELTA_BUFFER_MAX_BYTES: usize = 1024;
const ASSISTANT_DELTA_BUFFER_MAX_WAIT: Duration = Duration::from_millis(50);
const TERMINAL_TITLE_APP_NAME: &str = "LetCode";
const TERMINAL_TITLE_ACTIVE_FRAMES: [&str; 7] = [
    "Letcode", "lEtcode", "leTcode", "letCode", "letcOde", "letcoDe", "letcodE",
];
const TERMINAL_TITLE_TICKS_PER_FRAME: usize = 6;
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
    let app_name = match spinner_frame {
        Some(frame) => TERMINAL_TITLE_ACTIVE_FRAMES[frame % TERMINAL_TITLE_ACTIVE_FRAMES.len()],
        None => TERMINAL_TITLE_APP_NAME,
    };
    match session_title.filter(|title| !title.trim().is_empty()) {
        Some(title) => format!("{app_name}|{title}"),
        None => app_name.to_string(),
    }
}

fn assistant_delta_parts(
    event: &RunnerEvent,
) -> Option<(AssistantDeltaStream, Option<String>, String)> {
    match event {
        RunnerEvent::AssistantDelta(delta) => Some((
            AssistantDeltaStream {
                child_session_id: None,
                parent_tool_call_id: None,
                message_id: delta.message_id.clone(),
            },
            None,
            delta.delta.clone(),
        )),
        RunnerEvent::ChildAppEvent {
            child_session_id,
            agent_name,
            parent_tool_call_id,
            event: AppEvent::AssistantDelta(delta),
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
) -> RunnerEvent {
    let delta = match &stream.message_id {
        Some(message_id) => {
            crate::tui::events::AssistantDeltaEvent::with_message_id(message_id, delta)
        }
        None => crate::tui::events::AssistantDeltaEvent::new(delta),
    };
    match &stream.child_session_id {
        Some(child_session_id) => RunnerEvent::ChildAppEvent {
            child_session_id: child_session_id.clone(),
            agent_name: agent_name.clone(),
            parent_tool_call_id: stream.parent_tool_call_id.clone(),
            event: AppEvent::AssistantDelta(delta),
        },
        None => RunnerEvent::AssistantDelta(delta),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InterruptRequest {
    parent_tool_calls: Vec<(String, String)>,
    visible_child_session_id: Option<String>,
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
    pub context_window_tokens: Option<u64>,
    pub reasoning_effort: Option<ModelReasoningEffort>,
    pub reasoning_efforts: Vec<ModelReasoningEffort>,
}

impl AvailableModel {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            context_window_tokens: None,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
        }
    }

    pub fn with_context_window(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
    ) -> Self {
        Self {
            id: id.into(),
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
        Self {
            id: id.into(),
            label: label.into(),
            context_window_tokens,
            reasoning_effort,
            reasoning_efforts,
        }
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

#[derive(Debug, Default)]
pub struct NoopDrawer;

impl RuntimeDrawer for NoopDrawer {
    fn draw(&mut self, _state: &mut TuiState) -> io::Result<()> {
        Ok(())
    }
}

pub struct TuiRuntime {
    state: TuiState,
    runner_rx: mpsc::UnboundedReceiver<RunnerEvent>,
    permission_lifecycle: PermissionLifecycleController,
    pending_question_handle: Option<RunnerQuestionRequest>,
    pending_question_child_session_id: Option<String>,
    interrupt_confirmation_pending: bool,
    submitted_prompts: Vec<String>,
    queued_prompts: VecDeque<UserMessageSubmission>,
    queued_prompt_lifecycle: QueuedPromptLifecycle,
    runner_turn_active: bool,
    current_turn_output_tokens: u64,
    history_selection: Option<usize>,
    history_draft: Option<ComposerDraft>,
    available_models: Vec<AvailableModel>,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    assistant_delta_buffer: Option<AssistantDeltaBuffer>,
    session_title: Option<String>,
    spinner_frame: usize,
}

impl TuiRuntime {
    pub fn new(
        state: TuiState,
        runner_rx: mpsc::UnboundedReceiver<RunnerEvent>,
        available_models: Vec<AvailableModel>,
        sessions_dir: PathBuf,
        preferences_dir: PathBuf,
    ) -> Self {
        Self {
            state,
            runner_rx,
            permission_lifecycle: PermissionLifecycleController::default(),
            pending_question_handle: None,
            pending_question_child_session_id: None,
            interrupt_confirmation_pending: false,
            submitted_prompts: Vec::new(),
            queued_prompts: VecDeque::new(),
            queued_prompt_lifecycle: QueuedPromptLifecycle::default(),
            runner_turn_active: false,
            current_turn_output_tokens: 0,
            history_selection: None,
            history_draft: None,
            available_models,
            sessions_dir,
            preferences_dir,
            assistant_delta_buffer: None,
            session_title: None,
            spinner_frame: 0,
        }
    }

    pub fn state(&self) -> &TuiState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut TuiState {
        &mut self.state
    }

    fn terminal_title(&self) -> String {
        format_terminal_title(
            self.session_title.as_deref(),
            self.has_active_or_pending_runner_turn()
                .then_some(self.spinner_frame / TERMINAL_TITLE_TICKS_PER_FRAME),
        )
    }

    fn update_terminal_title(&self, terminal: &mut OwnedTerminal) -> io::Result<()> {
        terminal.set_title(&self.terminal_title())
    }

    fn show_toast(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.state.show_toast(message, kind);
    }

    pub fn submitted_prompts(&self) -> &[String] {
        &self.submitted_prompts
    }

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
        if self.pending_question_child_session_id.is_none() {
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

    pub fn into_state(self) -> TuiState {
        self.state
    }

    pub fn try_drain_runner_events(&mut self) {
        // Leave time in every frame for terminal input. In particular, an
        // unbounded stream of model deltas must not prevent a confirmed Esc
        // from reaching the runner.
        const MAX_RUNNER_EVENTS_PER_FRAME: usize = 256;
        self.flush_assistant_delta_buffer_if_due();
        for _ in 0..MAX_RUNNER_EVENTS_PER_FRAME {
            let Ok(event) = self.runner_rx.try_recv() else {
                break;
            };
            self.consume_runner_event(event);
        }
    }

    fn consume_runner_event(&mut self, event: RunnerEvent) {
        if let Some((stream, agent_name, delta)) = assistant_delta_parts(&event) {
            self.buffer_assistant_delta(stream, agent_name, delta);
        } else {
            self.flush_assistant_delta_buffer();
            self.apply_runner_event(event);
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
            self.apply_runner_event(event);
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
            self.apply_runner_event(assistant_delta_event(
                &buffer.stream,
                &buffer.agent_name,
                buffer.delta,
            ));
        }
    }

    pub fn apply_runner_event(&mut self, event: RunnerEvent) {
        let mut suppress_app_event = false;

        match &event {
            RunnerEvent::QuestionRequested { request, handle } => {
                if self
                    .begin_pending_question(request.clone(), handle.clone(), None)
                    .is_err()
                {
                    let _ = handle.cancel("another interactive request is already pending");
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                    suppress_app_event = true;
                }
            }
            RunnerEvent::PermissionRequested { event, handle } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state
                        .show_toast("Question already pending", ToastKind::Info);
                    suppress_app_event = true;
                } else if let Err(handle) = self
                    .permission_lifecycle
                    .begin_parent(event.clone(), handle.clone())
                {
                    let _ = handle.deny();
                    self.state
                        .show_toast("Permission already pending", ToastKind::Info);
                    suppress_app_event = true;
                } else {
                    self.state.toast = None;
                }
            }
            RunnerEvent::ChildQuestionRequested {
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
                    suppress_app_event = true;
                }
            }
            RunnerEvent::ChildPermissionRequested {
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
                    self.state.apply_child_app_event_with_agent(
                        child_session_id,
                        agent_name.as_deref(),
                        parent_tool_call_id.as_deref(),
                        AppEvent::PermissionRequested(event.clone()),
                    );
                }
            }
            RunnerEvent::PermissionResolved(resolution) => {
                if self.pending_permission_matches_call(&resolution.call_id, None) {
                    self.permission_lifecycle.clear();
                }
            }
            RunnerEvent::Done => {
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
                self.runner_turn_active = false;
            }
            RunnerEvent::Error(_) => {
                self.interrupt_confirmation_pending = false;
                self.queued_prompt_lifecycle.record_error();
            }
            RunnerEvent::FastModeChanged { enabled } => {
                self.state.set_fast_mode_enabled(*enabled);
            }
            RunnerEvent::ModelChanged { model_id } => {
                self.apply_restored_model(model_id.clone());
                self.show_toast("Model updated", ToastKind::Success);
            }
            RunnerEvent::QueuedPromptAccepted { prompt } => {
                self.queued_prompt_lifecycle.accept(&prompt.id);
            }
            RunnerEvent::SessionTitleUpdated { session_id, title } => {
                if self.state.session_id.as_deref() == Some(session_id) {
                    self.session_title = Some(title.clone());
                }
            }
            RunnerEvent::Interrupted => {
                self.permission_lifecycle.clear_if_parent();
                let _ = self
                    .cancel_pending_question("question cancelled because the turn was interrupted");
                self.interrupt_confirmation_pending = false;
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.runner_turn_active = false;
                self.state.activate_all_queued_user_message_previews();
            }
            RunnerEvent::UserMessage(user_message) => {
                self.queued_prompt_lifecycle.clear_dispatch_ready();
                self.runner_turn_active = true;
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
                    suppress_app_event = self
                        .state
                        .activate_queued_user_message(&user_message.submission_id);
                }
            }
            RunnerEvent::AssistantDelta(_)
            | RunnerEvent::ReasoningDelta(_)
            | RunnerEvent::ToolPending(_)
            | RunnerEvent::ToolCancelled(_)
            | RunnerEvent::ToolStarted(_)
            | RunnerEvent::ToolOutputDelta(_) => {
                self.queued_prompt_lifecycle.clear_dispatch_ready();
            }
            RunnerEvent::TokenUsage(token_usage) => {
                let mut token_usage = token_usage.clone();
                if token_usage.output_tokens > 0 {
                    self.current_turn_output_tokens = self
                        .current_turn_output_tokens
                        .saturating_add(token_usage.output_tokens);
                }
                token_usage.output_tokens = self.current_turn_output_tokens;
                self.state.apply_event(AppEvent::TokenUsage(token_usage));
                suppress_app_event = true;
            }
            RunnerEvent::SessionTokenUsage(token_usage) => {
                // A committed manual compaction replaces the request snapshot.
                // Its local estimate has no provider response/cache accounting.
                self.current_turn_output_tokens = 0;
                self.state
                    .apply_event(AppEvent::TokenUsage(token_usage.clone()));
                suppress_app_event = true;
            }
            RunnerEvent::ToolBatchFinished => {
                if !self.queued_prompts.is_empty()
                    && !self.queued_prompt_lifecycle.has_inflight_handoff()
                {
                    self.queued_prompt_lifecycle.mark_dispatch_ready();
                }
            }
            RunnerEvent::SessionResumed {
                session_id,
                branch_id,
                messages,
                records,
                evidence_count,
                model_id,
                token_usage,
                runtime_context,
            } => {
                let message_count = messages.len();
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
                self.permission_lifecycle.clear_if_parent();
                self.queued_prompts.clear();
                self.queued_prompt_lifecycle.reset();
                self.runner_turn_active = false;
                self.current_turn_output_tokens = 0;
                self.state.timeline.remove_queued_user_message_previews();
                if let Some(model_id) = model_id {
                    self.apply_restored_model(model_id.clone());
                }
                self.state.set_current_context_branch(branch_id.clone());
                if let Some(token_usage) = token_usage {
                    self.state.set_token_usage(token_usage.clone().into());
                }
                self.state.show_toast("Session resumed", ToastKind::Info);
            }
            RunnerEvent::ContextBranchChanged { branch_id } => {
                self.state.set_current_context_branch(branch_id.clone());
            }
            RunnerEvent::ChildSessionViewed {
                parent_session_id,
                child_session_id,
                agent_name,
                index,
                total,
                pool_ordinal,
                records,
                runtime_context,
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
            RunnerEvent::SessionHistoryLoaded { entries } => {
                self.open_history_tree_dialog(entries);
            }
            RunnerEvent::SessionStarted {
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
                self.runner_turn_active = false;
                self.current_turn_output_tokens = 0;
                self.state.timeline.remove_queued_user_message_previews();
                // A newly created, still-empty session remains on the dashboard.
                self.state.active_session = false;
                self.state
                    .set_current_context_branch(crate::transcript::ROOT_CONTEXT_BRANCH_ID);
                self.state
                    .show_toast("New session started", ToastKind::Info);
            }
            RunnerEvent::McpToolsDiscovered(servers) => {
                self.state.set_mcp_servers(servers.clone());
                self.refresh_open_mcp_dialog();
            }
            RunnerEvent::McpServerUpdated(server) => {
                self.state.update_mcp_server(server.clone());
                self.state
                    .set_mcp_server_updating(server.name.clone(), false);
                self.refresh_open_mcp_dialog();
                self.refresh_open_mcp_tools_dialog(&server.name);
            }
            RunnerEvent::McpServerUpdating { name, updating } => {
                self.state.set_mcp_server_updating(name.clone(), *updating);
                self.refresh_open_mcp_dialog();
            }
            RunnerEvent::McpServerToolsUpdated { name, tools } => {
                self.state.set_mcp_server_tools(name.clone(), tools.clone());
                self.refresh_open_mcp_tools_dialog(name);
            }
            RunnerEvent::McpDiscoveryUnavailable(error) => {
                self.state.mark_mcp_discovery_unavailable(error.clone());
                self.refresh_open_mcp_dialog();
            }
            RunnerEvent::McpDiagnostic(message) => {
                self.show_toast(message.clone(), ToastKind::Error);
            }
            RunnerEvent::ChildAppEvent {
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
                        AppEvent::Error(_) | AppEvent::Done | AppEvent::Interrupted
                    )
                {
                    let _ = self.cancel_pending_question(
                        "question cancelled because the child session stopped",
                    );
                }
                if matches!(
                    event,
                    AppEvent::Error(_) | AppEvent::Done | AppEvent::Interrupted
                ) {
                    self.interrupt_confirmation_pending = false;
                }
                self.state.apply_child_app_event_with_agent(
                    child_session_id,
                    agent_name.as_deref(),
                    parent_tool_call_id.as_deref(),
                    event.clone(),
                );
            }
            _ => {}
        }

        self.reproject_pending_permission();

        if !suppress_app_event {
            if let Some(app_event) = event.app_event() {
                self.state.apply_event(app_event);
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

    fn pending_permission_belongs_to_parent(&self) -> bool {
        self.permission_lifecycle.belongs_to_parent()
    }

    fn child_event_clears_pending_permission(
        &self,
        child_session_id: &str,
        event: &AppEvent,
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
            InputAction::MouseClick => Ok(None),
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
                    Ok(None)
                } else {
                    Ok(Some(RuntimeCommand::ViewParent))
                }
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
                Ok(None)
            }
            InputAction::DialogAccept => self.handle_dialog_accept(),
            InputAction::DialogToggle => self.handle_mcp_toggle(),
            InputAction::DialogCancel => {
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
                self.state.apply_event(AppEvent::Quit);
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
                if let Err(error) = refresh_child_session_view(&self.sessions_dir, &mut self.state)
                {
                    self.state
                        .apply_event(AppEvent::Error(ErrorEvent::new(format!(
                            "Context projection failed: {error}"
                        ))));
                }
                self.state.apply_event(AppEvent::Tick);
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

    pub fn run<D: RuntimeDrawer>(
        &mut self,
        terminal: &mut OwnedTerminal,
        drawer: &mut D,
    ) -> io::Result<()> {
        self.try_drain_runner_events();
        self.update_terminal_title(terminal)?;
        self.draw(drawer)
    }

    fn has_active_or_pending_runner_turn(&self) -> bool {
        has_active_or_pending_runner_turn(active_turn_state(
            &self.state,
            self.runner_turn_active,
            self.queued_prompt_lifecycle.has_inflight_handoff(),
            self.permission_lifecycle.is_pending(),
        ))
    }

    fn history_navigation_is_unavailable(&self) -> bool {
        self.has_active_or_pending_runner_turn()
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
        let active_runner_turn = self.has_active_or_pending_runner_turn();
        let active_turn_command_allowed = matches!(
            &parsed_command,
            Ok(CommandIntent::Help
                | CommandIntent::ContextBrowse
                | CommandIntent::McpBrowse
                | CommandIntent::SkillBrowse
                | CommandIntent::ToolOutputSet(_)
                | CommandIntent::TranscriptScrollbarSet(_)
                | CommandIntent::Child(_)
                | CommandIntent::Parent)
        );
        if active_runner_turn && !active_turn_command_allowed {
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
        self.runner_turn_active = true;
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
        self.runner_turn_active = true;
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        Some(RuntimeCommand::SubmitPrompt(prompt))
    }

    fn handle_interrupt(&mut self) -> Result<Option<RuntimeCommand>> {
        if !self.has_active_or_pending_runner_turn() {
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

    fn build_interrupt_request(&self) -> InterruptRequest {
        build_interrupt_request(
            self.state.timeline.active_tool_calls(),
            self.state
                .child_view_metadata()
                .map(|metadata| metadata.child_session_id),
            self.state.child_view_has_live_stream(),
        )
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
                self.state.apply_event(AppEvent::Quit);
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::Help => {
                self.push_command_notice(help_summary());
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            CommandIntent::ModelShow => self.show_model_dialog(),
            CommandIntent::ReasoningShow => self.show_reasoning_dialog(),
            CommandIntent::PermissionShow => self.show_permission_dialog(),
            CommandIntent::ToolOutputSet(mode) => self.handle_tool_output_command(mode),
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
                self.runner_turn_active = true;
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
            SessionCommand::ResumeSession(session_id) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ResumeSession(session_id),
            ))),
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
                    Ok(Some(SubmittedCommand::LocalOnly))
                } else {
                    Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewParent)))
                }
            }
            SessionCommand::DelegateSubagent { agent_name, task } => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.runner_turn_active = true;
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
        let prefs = TuiPreferences {
            tool_output_expanded: self.state.tool_output_expanded,
            transcript_scrollbar_visible: self.state.transcript_scrollbar_visible,
        };
        if let Err(error) = prefs.save_to_dir(&self.preferences_dir) {
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
        let prefs = TuiPreferences {
            tool_output_expanded: self.state.tool_output_expanded,
            transcript_scrollbar_visible: self.state.transcript_scrollbar_visible,
        };
        if let Err(error) = prefs.save_to_dir(&self.preferences_dir) {
            self.state
                .show_toast("Transcript scrollbar", ToastKind::Info);
            return SubmittedCommand::LocalOnly;
        }
        self.state
            .show_toast("Transcript scrollbar", ToastKind::Info);
        SubmittedCommand::LocalOnly
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
                "solo",
                "Solo",
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
            "solo" => 2,
            _ => 1,
        };
        self.state.open_dialog(dialog);
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_model_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let items = self
            .available_models
            .iter()
            .map(|model| {
                DialogItem::new(
                    model.id.clone(),
                    model.label.clone(),
                    Some(model.id.clone()),
                )
            })
            .collect::<Vec<_>>();
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
        if self.runner_turn_active {
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
                self.state
                    .set_model(selected.id.clone(), selected.label.clone());
                let context_window_tokens = self
                    .available_models
                    .iter()
                    .find(|model| model.id == selected.id)
                    .and_then(|model| model.context_window_tokens);
                self.state.set_model_context_window(context_window_tokens);
                let reasoning_effort = self
                    .available_models
                    .iter()
                    .find(|model| model.id == selected.id)
                    .and_then(|model| model.reasoning_effort.clone());
                self.state
                    .set_reasoning_effort_label(Some(reasoning_effort_status_label(
                        reasoning_effort,
                    )));
                self.show_toast(
                    format!("Model updated · {}", selected.label),
                    ToastKind::Success,
                );
                Ok(Some(RuntimeCommand::SetModel(selected.id)))
            }
            DialogKind::PermissionPicker => {
                self.state.close_dialog();
                let mode = match selected.id.as_str() {
                    "safe" => PermissionMode::Safe,
                    "solo" => PermissionMode::Solo,
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

    fn handle_selection_start(&mut self, col: u16, row: u16) {
        // 落在 transcript 内容区外不开始选择；点击空白/spacer 也返回 None
        if let Some(anchor) = self.state.map_mouse_to_anchor(col, row) {
            self.state.text_selection = Some(super::state::TextSelection {
                start: anchor.clone(),
                end: anchor,
            });
            self.state.selection_in_progress = true;
            self.state.selection_last_mouse = Some((col, row));
        } else {
            // 在 transcript 外点击：清除现有选择，避免残留高亮
            self.state.text_selection = None;
            self.state.selection_in_progress = false;
            self.state.selection_last_mouse = None;
        }
    }

    fn handle_selection_drag(&mut self, col: u16, row: u16) {
        if !self.state.selection_in_progress {
            return;
        }
        self.state.selection_last_mouse = Some((col, row));
        if let Some(anchor) = self.state.map_mouse_to_anchor(col, row) {
            if let Some(selection) = &mut self.state.text_selection {
                selection.end = anchor;
            }
        }
    }

    fn handle_selection_end(&mut self, col: u16, row: u16) {
        self.handle_selection_drag(col, row);
        self.state.selection_in_progress = false;
        self.state.selection_last_mouse = None;
        // 抛弃零宽选择（单击未拖动），避免接管 Ctrl+C 复制语义且无视觉反馈
        if let Some(selection) = &self.state.text_selection {
            if selection.start == selection.end {
                self.state.text_selection = None;
            }
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

enum RunnerCommand {
    Prompt(UserMessageSubmission),
    DelegateSubagent {
        agent_name: String,
        task: String,
    },
    Compact,
    ShowHistoryTree,
    Undo,
    Redo,
    NavigateHistory {
        target_entry_id: String,
    },
    ViewChild {
        navigation: SharedChildNavigation,
        anchor_child_session_id: Option<String>,
    },
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    ToggleFastMode,
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    #[cfg(test)]
    InspectHistory(tokio::sync::oneshot::Sender<Vec<crate::request_builder::HistoryItem>>),
}

/// Map private runner transport commands that the session coordinator owns as idle work.
fn runner_command_as_idle_session_command(
    command: &RunnerCommand,
) -> Option<crate::session::SessionCommand> {
    match command {
        RunnerCommand::ShowHistoryTree => Some(crate::session::SessionCommand::ShowHistoryTree),
        RunnerCommand::Undo => Some(crate::session::SessionCommand::Undo),
        RunnerCommand::Redo => Some(crate::session::SessionCommand::Redo),
        RunnerCommand::NavigateHistory { target_entry_id } => {
            Some(crate::session::SessionCommand::NavigateHistory {
                target_entry_id: target_entry_id.clone(),
            })
        }
        RunnerCommand::SetPermissionMode(mode) => {
            Some(crate::session::SessionCommand::SetPermissionMode(*mode))
        }
        RunnerCommand::SetModel(model) => {
            Some(crate::session::SessionCommand::SetModel(model.clone()))
        }
        RunnerCommand::ToggleFastMode => Some(crate::session::SessionCommand::ToggleFastMode),
        RunnerCommand::SetReasoningEffort(effort) => Some(
            crate::session::SessionCommand::SetReasoningEffort(effort.clone()),
        ),
        RunnerCommand::ViewParent => Some(crate::session::SessionCommand::ViewParent),
        RunnerCommand::ViewChild {
            navigation,
            anchor_child_session_id,
        } => Some(crate::session::SessionCommand::ViewChild {
            navigation: *navigation,
            anchor_child_session_id: anchor_child_session_id.clone(),
        }),
        RunnerCommand::Prompt(_)
        | RunnerCommand::DelegateSubagent { .. }
        | RunnerCommand::Compact
        | RunnerCommand::ResumeSession(_)
        | RunnerCommand::NewSession
        | RunnerCommand::ToggleMcpServer(_) => None,
        #[cfg(test)]
        RunnerCommand::InspectHistory(_) => None,
    }
}

enum RunnerControl {
    Command(RunnerCommand),
    Interrupt(InterruptRequest),
}

enum ActiveRunnerOperation<T> {
    Interrupted(InterruptRequest),
    Completed(T),
    Command(Option<RunnerCommand>),
}

fn drain_queued_runner_controls(
    control_rx: &mut mpsc::UnboundedReceiver<RunnerControl>,
    deferred_commands: &mut VecDeque<RunnerCommand>,
) -> Option<InterruptRequest> {
    loop {
        match control_rx.try_recv() {
            Ok(RunnerControl::Command(command)) => deferred_commands.push_back(command),
            Ok(RunnerControl::Interrupt(interrupt)) => return Some(interrupt),
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                return None;
            }
        }
    }
}

async fn next_idle_runner_command(
    control_rx: &mut mpsc::UnboundedReceiver<RunnerControl>,
    deferred_commands: &mut VecDeque<RunnerCommand>,
) -> Option<RunnerCommand> {
    loop {
        if let Some(command) = deferred_commands.pop_front() {
            return Some(command);
        }

        match control_rx.recv().await? {
            RunnerControl::Command(command) => return Some(command),
            // An idle interrupt is stale only when it appears before the next
            // command in the FIFO stream.
            RunnerControl::Interrupt(_) => {}
        }
    }
}

async fn select_active_runner_operation<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<RunnerControl>,
    deferred_commands: &mut VecDeque<RunnerCommand>,
    mut operation: Pin<&mut F>,
) -> ActiveRunnerOperation<T>
where
    F: Future<Output = T> + ?Sized,
{
    loop {
        // Keep commands queued while looking through the FIFO stream for an
        // already-arrived interrupt. This retains cancellation priority for a
        // live operation without losing commands that preceded the interrupt.
        if let Some(interrupt) = drain_queued_runner_controls(control_rx, deferred_commands) {
            return ActiveRunnerOperation::Interrupted(interrupt);
        }

        if let Some(command) = deferred_commands.pop_front() {
            return ActiveRunnerOperation::Command(Some(command));
        }

        tokio::select! {
            biased;
            control = control_rx.recv() => match control {
                Some(RunnerControl::Interrupt(interrupt)) => {
                    return ActiveRunnerOperation::Interrupted(interrupt);
                }
                Some(RunnerControl::Command(command)) => deferred_commands.push_back(command),
                None => return ActiveRunnerOperation::Command(None),
            },
            result = operation.as_mut() => return ActiveRunnerOperation::Completed(result),
        }
    }
}

async fn select_manual_compaction_operation<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<RunnerControl>,
    deferred_commands: &mut VecDeque<RunnerCommand>,
    mut operation: Pin<&mut F>,
) -> Option<T>
where
    F: Future<Output = T> + ?Sized,
{
    loop {
        if drain_queued_runner_controls(control_rx, deferred_commands).is_some() {
            return None;
        }

        tokio::select! {
            biased;
            Some(control) = control_rx.recv() => match control {
                RunnerControl::Interrupt(_) => return None,
                RunnerControl::Command(command) => deferred_commands.push_back(command),
            },
            result = operation.as_mut() => return Some(result),
        }
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

fn missing_api_key_error(api_key_hint: &str) -> ErrorEvent {
    ErrorEvent::new(format!(
        "API key is not set for the active provider. {}",
        api_key_hint
    ))
}

fn send_missing_api_key_error(runner_tx: &mpsc::UnboundedSender<RunnerEvent>, api_key_hint: &str) {
    let _ = runner_tx.send(RunnerEvent::Error(missing_api_key_error(api_key_hint)));
    let _ = runner_tx.send(RunnerEvent::Done);
}

fn short_session_id(session_id: &str) -> &str {
    session_id.get(..12).unwrap_or(session_id)
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

    fn label(self) -> &'static str {
        match self {
            Self::Expanded => "expanded",
            Self::Truncated => "truncated",
        }
    }
}

fn active_context_snapshot(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<transcript_projection::SessionRestoreSnapshot> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder.current_context_branch_id().map(str::to_string),
        )
    };
    transcript_projection::build_session_context_snapshot(
        session_id,
        records,
        transcript_projection::SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
    )
}

fn current_runtime_context(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<RuntimeActiveContext> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder
                .current_context_branch_id()
                .unwrap_or(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
                .to_string(),
        )
    };
    runtime_context_from_records(&records, &session_id, Some(&branch_id))
}

fn record_manual_compaction_error(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    message: String,
) -> ErrorEvent {
    let message = match transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))
        .and_then(|mut recorder| recorder.record_error(message.clone()))
    {
        Ok(()) => message,
        Err(error) => {
            format!("{message} (additionally failed to record transcript error: {error})")
        }
    };
    ErrorEvent::new(message)
}

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

fn block_source_label(block: &crate::context_view::ContextBlock) -> &'static str {
    match block.source {
        crate::context_view::ContextBlockSource::TranscriptSpan { .. } => "Source",
        crate::context_view::ContextBlockSource::SummaryArtifact { .. } => "Summary",
    }
}

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

fn truncate_dialog_text(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= 120 {
        return collapsed;
    }
    let mut out = collapsed.chars().take(120).collect::<String>();
    out.push('…');
    out
}

fn sessions_dir_for_transcript(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Result<PathBuf> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    recorder
        .path()
        .parent()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("transcript path has no parent directory"))
}

fn refresh_child_session_view(
    sessions_dir: &std::path::Path,
    state: &mut TuiState,
) -> Result<bool> {
    let Some(metadata) = state.child_view_metadata() else {
        return Ok(false);
    };

    let parent_records = crate::transcript::read_records(
        sessions_dir.join(format!("{}.jsonl", metadata.parent_session_id)),
    )?;
    let children = SubagentPool::child_sessions(sessions_dir, &parent_records);
    let completed_position = children
        .iter()
        .position(|child| child.child_session_id == metadata.child_session_id);

    let records =
        read_child_session_records_allow_partial_tail(sessions_dir, &metadata.child_session_id)?;
    let next_index = completed_position.unwrap_or(metadata.index);
    let next_total = if completed_position.is_some() {
        children.len()
    } else {
        metadata.total
    };

    if metadata.record_count == records.len()
        && metadata.index == next_index
        && metadata.total == next_total
    {
        return Ok(false);
    }

    if state.child_view_has_live_stream() {
        return Ok(false);
    }

    // A refresh is a scope replacement, not a raw presentation update. Build
    // the canonical payload before touching the visible child state.
    let context = runtime_context_from_records(&records, &metadata.child_session_id, None)?;
    if completed_position.is_some() {
        state.try_replace_child_timeline_from_records_with_runtime_context(
            &records,
            metadata.parent_session_id,
            metadata.child_session_id,
            metadata.agent_name,
            next_index,
            next_total,
            metadata.pool_ordinal,
            context,
        )?;
    } else {
        state.try_refresh_child_timeline_from_records_with_runtime_context(&records, context)?;
    }
    Ok(true)
}

fn send_subagent_interrupted(
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    child_session_id: Option<String>,
) {
    if let Some(child_session_id) = child_session_id {
        let _ = runner_tx.send(RunnerEvent::ChildAppEvent {
            child_session_id,
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::Interrupted,
        });
    }
    let _ = runner_tx.send(RunnerEvent::Interrupted);
}

fn record_interrupt_transcript(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    interrupt: &InterruptRequest,
) {
    let mut recorder = match transcript.lock() {
        Ok(recorder) => recorder,
        Err(_) => return,
    };
    let cursor = transcript_projection::SessionContextCursor {
        branch_id: Some(
            recorder
                .current_context_branch_id()
                .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
                .to_string(),
        ),
        leaf_sequence: None,
    };

    let turn_id = match read_records(recorder.path()).and_then(|records| {
        transcript_projection::active_turn_id_at_context_cursor(records, cursor)
    }) {
        Ok(turn_id) => turn_id,
        Err(_) => return,
    };

    for (call_id, name) in &interrupt.parent_tool_calls {
        let _ = recorder.record_tool_call_cancelled(call_id.clone(), name.clone());
    }

    if let Some(turn_id) = turn_id {
        let _ = recorder.record_turn_interrupted(Some(turn_id));
    }
}

fn rehydrate_agent_from_transcript<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<()>
where
    C: Config,
{
    let path = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();
    let records = read_records(&path)?;
    let session_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid transcript path: {}", path.display()))?
        .to_string();
    let snapshot = project_runtime_restore_snapshot_with_children(
        &session_id,
        records,
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &sessions_dir_for_transcript(transcript)?,
    )?;
    let branch_id = snapshot.branch_id.clone();
    let max_turn_id = snapshot.max_turn_id;
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    agent.restore_runtime_snapshot(snapshot.protocol_frames, snapshot.snapshot)?;
    agent.restore_turn_sequence(max_turn_id);
    sync_recorder_branch(&mut recorder, &branch_id);
    Ok(())
}

fn manual_compaction_session_token_usage<C>(agent: &Agent<C>) -> Result<TokenUsageEvent>
where
    C: Config,
{
    let usage = agent.session_token_usage()?;
    Ok(TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    ))
}

async fn run_manual_compaction<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    control_rx: &mut mpsc::UnboundedReceiver<RunnerControl>,
    deferred_commands: &mut VecDeque<RunnerCommand>,
) where
    C: Config + Clone,
{
    let transcript = Arc::clone(transcript);
    let snapshot_transcript = Arc::clone(&transcript);
    agent.set_runtime_snapshot_provider(Arc::new(move || {
        let transcript = snapshot_transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        let records = read_records(transcript.path())?;
        Ok(
            crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                transcript.session_id().to_string(),
                records,
                crate::transcript::transcript_projection::SessionContextCursor {
                    branch_id: transcript.current_context_branch_id().map(str::to_string),
                    leaf_sequence: None,
                },
                &[],
            )?
            .snapshot,
        )
    }));
    let event_transcript = Arc::clone(&transcript);
    let event_runner_tx = runner_tx.clone();
    // Persistence is the compaction transaction boundary. A cancellation that
    // arrives after it must retain the record.
    let compaction_persisted = Arc::new(AtomicBool::new(false));
    let event_compaction_persisted = Arc::clone(&compaction_persisted);
    let on_event = move |event| {
        let transcript = Arc::clone(&event_transcript);
        let runner_tx = event_runner_tx.clone();
        let compaction_persisted = Arc::clone(&event_compaction_persisted);
        async move {
            match event {
                AgentEvent::ContextCompactionStarted { .. } => {
                    let _ = runner_tx.send(RunnerEvent::CompactionStarted);
                }
                AgentEvent::ContextCompactionNoProgress(no_progress) => {
                    let _ = runner_tx.send(RunnerEvent::CompactionNoProgress {
                        blockers: no_progress
                            .blockers
                            .into_iter()
                            .map(|blocker| blocker.label().to_string())
                            .collect(),
                    });
                }
                AgentEvent::ContextCompactionFailed { .. } => {
                    let _ = runner_tx.send(RunnerEvent::CompactionFailed);
                }
                AgentEvent::ContextCompacted(event) => {
                    let summary = event.summary.clone();
                    let mut recorder = transcript
                        .lock()
                        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
                    persist_agent_event(&mut recorder, &AgentEvent::ContextCompacted(event))?;
                    drop(recorder);
                    compaction_persisted.store(true, Ordering::Release);
                    // Persistence acknowledges the transaction. A closed
                    // presentation channel cannot roll it back.
                    let _ = runner_tx.send(RunnerEvent::CompactionCommitted {
                        summary: Some(summary),
                    });
                }
                AgentEvent::ContextCompactionDelta { delta } => {
                    let _ = runner_tx.send(RunnerEvent::CompactionPreviewDelta { delta });
                }
                _ => {}
            }
            Ok(())
        }
    };
    let mut on_start = || Ok(());
    let mut on_delta = |_delta: &str| Ok(());
    // Drop the compaction future before reporting cancellation so a late
    // durable acknowledgement from a cancelled attempt cannot reach the UI.
    let compaction_result = {
        let compact = agent.compact_session_stream_async(on_event, &mut on_start, &mut on_delta);
        tokio::pin!(compact);
        select_manual_compaction_operation(control_rx, deferred_commands, compact.as_mut()).await
    };

    match compaction_result {
        None => {
            // Manual compaction is not a model turn: do not write
            // TurnInterrupted. Restore the mutable agent from durable state so
            // the next command starts cleanly.
            let rehydrated = match rehydrate_agent_from_transcript(agent, &transcript) {
                Ok(()) => true,
                Err(error) => {
                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                        "failed to restore cancelled compaction context: {error}"
                    ))));
                    false
                }
            };
            if compaction_persisted.load(Ordering::Acquire) {
                // The durable callback won before cancellation. The candidate
                // may not have been installed in memory yet, so rehydration is
                // authoritative.
                if rehydrated {
                    match manual_compaction_session_token_usage(agent) {
                        Ok(token_usage) => {
                            let _ = runner_tx.send(RunnerEvent::SessionTokenUsage(token_usage));
                        }
                        Err(error) => {
                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                "failed to refresh committed compacted token usage: {error}"
                            ))));
                        }
                    }
                }
                match current_runtime_context(&transcript) {
                    Ok(context) => {
                        let _ = runner_tx.send(RunnerEvent::RuntimeContextUpdated(
                            RuntimeContextUpdatedEvent {
                                context,
                                disposition: RuntimeContextDisposition::Advance,
                            },
                        ));
                    }
                    Err(error) => {
                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                            "failed to refresh committed compacted context: {error}"
                        ))));
                    }
                }
            } else {
                let _ = runner_tx.send(RunnerEvent::CompactionFailed);
            }
        }
        Some(Ok(ManualCompactionOutcome::Compacted { .. })) => {
            match manual_compaction_session_token_usage(agent) {
                Ok(token_usage) => {
                    let _ = runner_tx.send(RunnerEvent::SessionTokenUsage(token_usage));
                }
                Err(error) => {
                    let event = record_manual_compaction_error(
                        &transcript,
                        format!("failed to refresh compacted token usage: {error}"),
                    );
                    let _ = runner_tx.send(RunnerEvent::Error(event));
                }
            }
            match current_runtime_context(&transcript) {
                Ok(context) => {
                    let _ = runner_tx.send(RunnerEvent::RuntimeContextUpdated(
                        RuntimeContextUpdatedEvent {
                            context,
                            disposition: RuntimeContextDisposition::Advance,
                        },
                    ));
                }
                Err(error) => {
                    let event = record_manual_compaction_error(
                        &transcript,
                        format!("failed to refresh compacted context: {error}"),
                    );
                    let _ = runner_tx.send(RunnerEvent::Error(event));
                }
            }
        }
        Some(Ok(ManualCompactionOutcome::NoProgress(_))) => {}
        Some(Err(error)) => {
            let event = record_manual_compaction_error(
                &transcript,
                format!("failed to compact context: {error}"),
            );
            let _ = runner_tx.send(RunnerEvent::Error(event));
        }
    }

    let _ = runner_tx.send(RunnerEvent::Done);
}

fn initial_session_metadata(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<(String, Option<String>)> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    Ok((
        recorder.session_id().to_string(),
        session_title_from_records(&read_records(recorder.path())?),
    ))
}

pub async fn run_tui<C>(
    mut agent: Agent<C>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    api_key_configured: bool,
    api_key_hint: String,
    provider_label: String,
    available_models: Vec<AvailableModel>,
    startup_toast: Option<StartupToast>,
    skill_cards: Vec<SkillCard>,
    mcp_config_path: PathBuf,
    mcp_config: indexmap::IndexMap<String, crate::config::McpServerConfig>,
    mcp_tools_rx: Option<mpsc::UnboundedReceiver<Vec<mcp::McpServerDiscovery>>>,
) -> Result<()>
where
    C: Config + Clone + Send + Sync + 'static,
{
    {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        crate::session::sync_agent_context_scope_from_recorder(&mut agent, &recorder)?;
    }
    let model_id = agent.model().to_string();
    let model_label = agent.model().to_string();
    let permission_mode_label = agent.permission_mode().to_string();
    let mut state = TuiState::new(model_id, model_label, permission_mode_label);
    let (session_id, session_title) = initial_session_metadata(&transcript)?;
    state.session_id = Some(session_id);
    state.set_skill_cards(skill_cards);
    let preferences = TuiPreferences::load_from_dir(&preferences_dir);
    state.set_tool_output_expanded(preferences.tool_output_expanded);
    state.set_transcript_scrollbar_visible(preferences.transcript_scrollbar_visible);
    state.set_provider_label(provider_label);
    state.set_fast_mode_enabled(agent.fast_mode_enabled());

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

    if !api_key_configured {
        state.show_toast("Missing API key", ToastKind::Info);
    }

    if let Some(toast) = startup_toast {
        state.show_toast(toast.message, toast.kind);
    }

    let (runner_tx, runner_rx) = mpsc::unbounded_channel();
    let (control_tx, mut control_rx) = mpsc::unbounded_channel::<RunnerControl>();
    let subagent_runtime = SubagentPool::new();
    let cleanup_subagent_runtime = subagent_runtime.clone();
    let mut runtime = TuiRuntime::new(
        state,
        runner_rx,
        available_models,
        sessions_dir.clone(),
        preferences_dir,
    );
    runtime.session_title = session_title;
    let mut terminal = OwnedTerminal::new()?;
    runtime.update_terminal_title(&mut terminal)?;
    let mut drawer = TerminalDrawer::new(&mut terminal);

    let cleanup_transcript = Arc::clone(&transcript);
    let runner_transcript = Arc::clone(&transcript);
    let runner_task = tokio::spawn(async move {
        let transcript = runner_transcript;
        let runner: AgentRunner<C> =
            AgentRunner::with_transcript(runner_tx.clone(), transcript.clone())
                .with_subagent_runtime(subagent_runtime.clone(), sessions_dir.clone());
        let mut agent = agent;
        let mut mcp_tools_rx = mcp_tools_rx;
        let mut mcp_config = mcp_config;
        let mut mcp_registered_tools: HashMap<String, Vec<String>> = HashMap::new();
        let subagent_runtime = subagent_runtime;
        let mut deferred_commands = VecDeque::new();

        loop {
            tokio::select! {
                biased;
                command = next_idle_runner_command(&mut control_rx, &mut deferred_commands) => {
                    let Some(command) = command else {
                        break;
                    };

                    if let Some(session_command) = runner_command_as_idle_session_command(&command) {
                        let _ = crate::session::SessionCoordinator::dispatch_idle_command(
                            session_command,
                            &mut agent,
                            &transcript,
                            &runner_tx,
                            Some(sessions_dir.as_path()),
                        );
                        continue;
                    }

                    let prompt = match command {
                        RunnerCommand::ToggleMcpServer(server_name) => {
                            let Some(server_config) = mcp_config.get(&server_name).cloned() else {
                                let _ = runner_tx.send(RunnerEvent::McpServerUpdating {
                                    name: server_name.clone(),
                                    updating: false,
                                });
                                let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                    "MCP server '{server_name}' is no longer configured"
                                )));
                                continue;
                            };
                            let enabled = !server_config.enabled;
                            let persisted_config = match crate::config::persist_mcp_server_enabled(
                                &mcp_config_path,
                                &server_name,
                                enabled,
                            ) {
                                Ok(config) => config,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::McpServerUpdating {
                                        name: server_name.clone(),
                                        updating: false,
                                    });
                                    let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                        "failed to persist MCP server '{server_name}': {error}"
                                    )));
                                    continue;
                                }
                            };
                            mcp_config.insert(server_name.clone(), persisted_config);
                            if !enabled {
                                for tool_name in mcp_registered_tools
                                    .remove(&server_name)
                                    .unwrap_or_default()
                                {
                                    agent.unregister_tool(&tool_name);
                                }
                                let _ = runner_tx.send(RunnerEvent::McpServerUpdated(
                                    mcp::McpServerCatalogEntry {
                                        name: server_name,
                                        enabled: false,
                                        status: mcp::McpServerStatus::Disabled,
                                    },
                                ));
                                continue;
                            }

                            let mut one_server = indexmap::IndexMap::new();
                            one_server.insert(
                                server_name.clone(),
                                mcp_config
                                    .get(&server_name)
                                    .expect("configured MCP server should remain present")
                                    .clone(),
                            );
                            let discovery = mcp::discover_servers(&one_server)
                                .await
                                .into_iter()
                                .next()
                                .expect("single MCP server discovery should return one result");
                            let mut server = discovery.server;
                            let mut catalog_tools = Vec::new();
                            match server.status {
                                mcp::McpServerStatus::Online { .. } => {
                                    let mut registered = Vec::new();
                                    for tool in discovery.tools {
                                        let tool_name = tool.name().to_string();
                                        let catalog_entry = tool.catalog_entry();
                                        if let Err(error) = agent.try_register_tool(tool) {
                                            let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                                "failed to register MCP tool '{tool_name}': {error}"
                                            )));
                                        } else {
                                            registered.push(tool_name);
                                            catalog_tools.push(catalog_entry);
                                        }
                                    }
                                    server.status = mcp::McpServerStatus::Online {
                                        tool_count: registered.len(),
                                    };
                                    mcp_registered_tools.insert(server_name, registered);
                                }
                                mcp::McpServerStatus::Offline { ref message } => {
                                    let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                        "MCP server '{}' is offline: {message}",
                                        server.name
                                    )));
                                }
                                mcp::McpServerStatus::Disabled => unreachable!("enabled server was discovered"),
                            }
                            let _ = runner_tx.send(RunnerEvent::McpServerToolsUpdated {
                                name: server.name.clone(),
                                tools: catalog_tools,
                            });
                            let _ = runner_tx.send(RunnerEvent::McpServerUpdated(server));
                            continue;
                        }
                        RunnerCommand::Prompt(prompt) => prompt,
                        RunnerCommand::ShowHistoryTree
                        | RunnerCommand::Undo
                        | RunnerCommand::Redo
                        | RunnerCommand::NavigateHistory { .. }
                        | RunnerCommand::SetPermissionMode(_)
                        | RunnerCommand::SetModel(_)
                        | RunnerCommand::ToggleFastMode
                        | RunnerCommand::SetReasoningEffort(_)
                        | RunnerCommand::ViewChild { .. }
                        | RunnerCommand::ViewParent => {
                            // Idle commands are handled above via SessionCoordinator.
                            continue;
                        }
                        RunnerCommand::DelegateSubagent { agent_name, task } => {
                            if !api_key_configured {
                                send_missing_api_key_error(&runner_tx, &api_key_hint);
                                continue;
                            }

                            let parent_session_id = match transcript.lock() {
                                Ok(recorder) => recorder.session_id().to_string(),
                                Err(_) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        "transcript recorder poisoned",
                                    )));
                                    let _ = runner_tx.send(RunnerEvent::Done);
                                    continue;
                                }
                            };

                            let invocation = match normalize_subagent_input(
                                &format!("agent__{agent_name}"),
                                &json!({ "task": task }),
                            ) {
                                Ok(input) => SubagentInvocation {
                                    prompt: input.objective.clone(),
                                    input,
                                    parent_tool_call_id: None,
                                },
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        format!("{error:#}"),
                                    )));
                                    let _ = runner_tx.send(RunnerEvent::Done);
                                    continue;
                                }
                            };

                            let (interrupted, child_started, interrupted_child_session_id) = {
                                let delegate = subagent_runtime.run_named_governed(
                                    &agent,
                                    &agent_name,
                                    invocation,
                                    sessions_dir.clone(),
                                    parent_session_id,
                                    format!(
                                        "turn-{}",
                                        std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_millis()
                                    ),
                                    Some(transcript.clone()),
                                    Some(crate::session::subagent_event_sender::<C>(
                                        runner_tx.clone(),
                                    )),
                                );

                                tokio::pin!(delegate);
                                let mut interrupted = false;
                                let mut child_started = false;
                                let mut interrupted_child_session_id = None;

                                loop {
                                    match select_active_runner_operation(
                                        &mut control_rx,
                                        &mut deferred_commands,
                                        delegate.as_mut(),
                                    )
                                    .await
                                    {
                                        ActiveRunnerOperation::Interrupted(interrupt) => {
                                            child_started = subagent_runtime.is_running();
                                            interrupted = true;
                                            interrupted_child_session_id = subagent_runtime
                                                .active_child()
                                                .map(|child| child.child_session_id);
                                            if child_started {
                                                subagent_runtime.cancel_active();
                                            }
                                            record_interrupt_transcript(&transcript, &interrupt);
                                            if child_started {
                                                if let Err(error) = delegate.await {
                                                    let _ = runner_tx.send(RunnerEvent::Error(
                                                        ErrorEvent::new(format!("{error:#}")),
                                                    ));
                                                }
                                            }
                                            break;
                                        }
                                        ActiveRunnerOperation::Completed(result) => {
                                            match result {
                                                Ok(_) => {
                                                    let _ = runner_tx.send(RunnerEvent::Done);
                                                }
                                                Err(error) => {
                                                    let _ = runner_tx.send(RunnerEvent::Error(
                                                        ErrorEvent::new(format!("{error:#}")),
                                                    ));
                                                    let _ = runner_tx.send(RunnerEvent::Done);
                                                }
                                            }
                                            break;
                                        }
                                        ActiveRunnerOperation::Command(Some(
                                            RunnerCommand::Prompt(prompt),
                                        )) => {
                                            deferred_commands.push_front(RunnerCommand::Prompt(prompt));
                                            let _ = runner_tx.send(RunnerEvent::AssistantDone {
                                                message_id: None,
                                            });
                                            break;
                                        }
                                        ActiveRunnerOperation::Command(Some(
                                            RunnerCommand::ViewChild {
                                                navigation,
                                                anchor_child_session_id,
                                            },
                                        )) => {
                                            crate::session::SessionCoordinator::emit_view_child(
                                                &transcript,
                                                &runner_tx,
                                                Some(sessions_dir.as_path()),
                                                navigation,
                                                anchor_child_session_id.as_deref(),
                                            );
                                        }
                                        ActiveRunnerOperation::Command(Some(
                                            RunnerCommand::ViewParent,
                                        )) => {
                                            crate::session::SessionCoordinator::emit_view_parent(
                                                &transcript,
                                                &runner_tx,
                                                Some(sessions_dir.as_path()),
                                            );
                                        }
                                        ActiveRunnerOperation::Command(Some(
                                            RunnerCommand::Undo | RunnerCommand::Redo,
                                        )) => {
                                            let _ = runner_tx.send(RunnerEvent::Notice(
                                                NoticeEvent::info(
                                                    "history navigation is unavailable while a turn is active",
                                                ),
                                            ));
                                        }
                                        ActiveRunnerOperation::Command(Some(
                                            RunnerCommand::ShowHistoryTree
                                            | RunnerCommand::NavigateHistory { .. },
                                        )) => {
                                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                                "history navigation is unavailable while a turn is active",
                                            )));
                                        }
                                        ActiveRunnerOperation::Command(Some(_)) => {}
                                        ActiveRunnerOperation::Command(None) => break,
                                    }
                                }

                                (interrupted, child_started, interrupted_child_session_id)
                            };

                            if interrupted {
                                if child_started {
                                    if let Err(error) =
                                        rehydrate_agent_from_transcript(&mut agent, &transcript)
                                    {
                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                            "failed to restore interrupted session context: {error}"
                                        ))));
                                    }
                                }
                                send_subagent_interrupted(&runner_tx, interrupted_child_session_id);
                            }
                            continue;
                        }
                        RunnerCommand::Compact => {
                            if !api_key_configured {
                                send_missing_api_key_error(&runner_tx, &api_key_hint);
                                let _ = runner_tx.send(RunnerEvent::Done);
                                continue;
                            }
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                                    "Wait for the active subagent to finish before compacting context",
                                )));
                                let _ = runner_tx.send(RunnerEvent::Done);
                                continue;
                            }

                            run_manual_compaction(
                                &mut agent,
                                &transcript,
                                &runner_tx,
                                &mut control_rx,
                                &mut deferred_commands,
                            )
                            .await;
                            continue;
                        }
                        #[cfg(test)]
                        RunnerCommand::InspectHistory(reply) => {
                            let _ = reply.send(agent.history_for_test().to_vec());
                            continue;
                        }
                        RunnerCommand::ResumeSession(prefix) => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                                    "Wait for the active subagent to finish before resuming another session",
                                )));
                                continue;
                            }

                            let session_id = match crate::session::resolve_session_prefix(
                                &sessions_dir,
                                &prefix,
                            ) {
                                Ok(session_id) => session_id,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        error.to_string(),
                                    )));
                                    continue;
                                }
                            };
                            let prepared = match crate::session::prepare_resume_package(
                                &sessions_dir,
                                session_id,
                            ) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to prepare resume: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let runtime_context =
                                match RuntimeActiveContext::try_from(&prepared.snapshot.snapshot) {
                                Ok(context) => context,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to validate restored session context: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let target_model = prepared
                                .snapshot
                                .latest_model
                                .as_deref()
                                .unwrap_or(agent.model());
                            let token_usage = match restored_session_token_usage(
                                &agent,
                                target_model,
                                &prepared.snapshot.snapshot,
                            ) {
                                Ok(usage) => Some(usage),
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore session token usage: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let resumed_event =
                                session_resumed_event(&prepared, runtime_context, token_usage);
                            let fast_mode_auto_disabled = match crate::session::install_prepared_resume_for_agent(
                                &mut agent,
                                &transcript,
                                prepared,
                            ) {
                                Ok(auto_disabled) => auto_disabled,
                                Err(error) => {
                                    if error.fast_mode_auto_disabled {
                                        let _ = runner_tx.send(RunnerEvent::FastModeChanged { enabled: false });
                                        let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                                            "Fast mode auto-disabled: current model is unavailable",
                                        )));
                                    }
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to install resumed session: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            if fast_mode_auto_disabled {
                                let _ = runner_tx.send(RunnerEvent::FastModeChanged { enabled: false });
                                let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                                    "Fast mode auto-disabled: current model is unavailable",
                                )));
                            }
                            let _ = runner_tx.send(resumed_event);
                            continue;
                        }
                        RunnerCommand::NewSession => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                                    "Wait for the active subagent to finish before starting a new session",
                                )));
                                continue;
                            }

                            let prepared = match crate::session::prepare_new_session_package(
                                &sessions_dir,
                                agent.model().to_string(),
                            ) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to create session transcript: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let started_event = session_started_event(&prepared);
                            let new_path = prepared.recorder.path().to_path_buf();
                            if let Err(error) =
                                crate::session::install_prepared_new_session_for_agent(
                                    &mut agent,
                                    &transcript,
                                    prepared,
                                )
                            {
                                let _ = remove_empty_session_file(&new_path);
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to install new session: {error}"
                                ))));
                                continue;
                            }
                            let _ = runner_tx.send(started_event);
                            continue;
                        }
                    };

                    let _ = runner_tx.send(RunnerEvent::QueuedPromptAccepted {
                        prompt: prompt.clone(),
                    });

                    if !api_key_configured {
                        send_missing_api_key_error(&runner_tx, &api_key_hint);
                        continue;
                    }

                    let interrupted = {
                        let run = runner.run_prompt(&mut agent, prompt);
                        tokio::pin!(run);
                        let mut interrupted = None;

                        loop {
                            match select_active_runner_operation(
                                &mut control_rx,
                                &mut deferred_commands,
                                run.as_mut(),
                            )
                            .await
                            {
                                ActiveRunnerOperation::Interrupted(interrupt) => {
                                    interrupted = Some(interrupt);
                                    break;
                                }
                                ActiveRunnerOperation::Completed(_) => break,
                                ActiveRunnerOperation::Command(command) => match command {
                                    Some(RunnerCommand::Prompt(prompt)) => {
                                        deferred_commands.push_front(RunnerCommand::Prompt(prompt));
                                        let _ = runner_tx.send(RunnerEvent::AssistantDone {
                                            message_id: None,
                                        });
                                        break;
                                    }
                                    Some(RunnerCommand::ViewChild {
                                        navigation,
                                        anchor_child_session_id,
                                    }) => {
                                        crate::session::SessionCoordinator::emit_view_child(
                                            &transcript,
                                            &runner_tx,
                                            Some(sessions_dir.as_path()),
                                            navigation,
                                            anchor_child_session_id.as_deref(),
                                        );
                                    }
                                    Some(RunnerCommand::ViewParent) => {
                                        crate::session::SessionCoordinator::emit_view_parent(
                                            &transcript,
                                            &runner_tx,
                                            Some(sessions_dir.as_path()),
                                        );
                                    }
                                    Some(RunnerCommand::Undo) | Some(RunnerCommand::Redo) => {
                                        let _ = runner_tx.send(RunnerEvent::Notice(
                                            NoticeEvent::info(
                                                "history navigation is unavailable while a turn is active",
                                            ),
                                        ));
                                    }
                                    Some(RunnerCommand::ShowHistoryTree)
                                    | Some(RunnerCommand::NavigateHistory { .. }) => {
                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                            "history navigation is unavailable while a turn is active",
                                        )));
                                    }
                                    Some(_) => {
                                        let _ = runner_tx.send(RunnerEvent::Notice(
                                            NoticeEvent::info("Turn still running · navigation only"),
                                        ));
                                    }
                                    None => break,
                                },
                            }
                        }

                        interrupted
                    };

                    if let Some(interrupt) = interrupted {
                        subagent_runtime.cancel_active();
                        record_interrupt_transcript(&transcript, &interrupt);
                        if let Err(error) = rehydrate_agent_from_transcript(&mut agent, &transcript) {
                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                "failed to restore interrupted session context: {error}"
                            ))));
                        }
                        send_subagent_interrupted(&runner_tx, interrupt.visible_child_session_id);
                    }
                }
                discovery = async {
                    mcp_tools_rx
                        .as_mut()
                        .expect("MCP discovery receiver should exist when select branch is enabled")
                        .recv()
                        .await
                }, if mcp_tools_rx.is_some() => {
                    let Some(discovery) = discovery else {
                        mcp_tools_rx = None;
                        continue;
                    };
                    mcp_tools_rx = None;

                    let mut servers = Vec::with_capacity(discovery.len());
                    for server_discovery in discovery {
                        let mut server = server_discovery.server;
                        let mut catalog_tools = Vec::new();
                        if let mcp::McpServerStatus::Offline { message } = &server.status {
                            let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                "MCP server '{}' is offline: {message}",
                                server.name
                            )));
                        }
                        let mut registered = Vec::new();
                        for tool in server_discovery.tools {
                            let tool_name = tool.name().to_string();
                            let catalog_entry = tool.catalog_entry();
                            if let Err(error) = agent.try_register_tool(tool) {
                                let _ = runner_tx.send(RunnerEvent::McpDiagnostic(format!(
                                    "failed to register MCP tool '{tool_name}': {error}"
                                )));
                            } else {
                                registered.push(tool_name);
                                catalog_tools.push(catalog_entry);
                            }
                        }
                        if matches!(server.status, mcp::McpServerStatus::Online { .. }) {
                            server.status = mcp::McpServerStatus::Online {
                                tool_count: registered.len(),
                            };
                            mcp_registered_tools.insert(server.name.clone(), registered);
                        }
                        let _ = runner_tx.send(RunnerEvent::McpServerToolsUpdated {
                            name: server.name.clone(),
                            tools: catalog_tools,
                        });
                        servers.push(server);
                    }
                    let _ = runner_tx.send(RunnerEvent::McpToolsDiscovered(servers));
                }
            }
        }
    });

    loop {
        runtime.try_drain_runner_events();
        if let Some(command) = runtime.take_next_queued_prompt_command() {
            command_dispatch::dispatch_command(&mut runtime, command, &control_tx, true);
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
                            &control_tx,
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
                            &control_tx,
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
                            &control_tx,
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

    drop(control_tx);
    cleanup_subagent_runtime.cancel_active();
    runner_task.abort();
    let _ = runner_task.await;
    remove_current_empty_session(&cleanup_transcript)?;

    Ok(())
}

struct TerminalDrawer<'a> {
    terminal: &'a mut OwnedTerminal,
}

impl<'a> TerminalDrawer<'a> {
    fn new(terminal: &'a mut OwnedTerminal) -> Self {
        Self { terminal }
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
        terminal.draw(|frame| render::render(frame, state))?;
        let _ = terminal.hide_cursor();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{
        AutoContinueState, CacheUsageReport, TodoItem, TodoStatus, TokenUsageEstimate,
        TurnFinalizedEvent, TurnStartedEvent,
    };
    use crate::config::CompactionConfig;
    use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
    };
    use crate::request_builder::{HistoryItem, ModelRequestMetadata};
    use crate::transcript::{
        ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord, TranscriptRecorder, read_records,
    };
    use crate::tui::{
        AppEvent, AppPhase, AssistantDeltaEvent, PermissionDecision, PermissionRequestEvent,
        PermissionResolutionEvent, PermissionResponse, RunnerEvent, RunnerPermissionRequest,
        TimelineItem, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
    };
    use async_openai::{Client, config::OpenAIConfig};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::{Notify, mpsc, oneshot};
    use tokio::task::JoinHandle;
    use tokio::time::timeout;

    fn event_context(session_id: &str, leaf_sequence: u64) -> RuntimeActiveContext {
        let mut snapshot =
            crate::runtime_context::RuntimeSnapshot::new(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
                .with_session_id(session_id)
                .with_leaf_sequence(leaf_sequence);
        snapshot.active_context.active_node_id = snapshot
            .context_tree
            .active_node_id()
            .map(|node_id| node_id.as_str().to_string());
        RuntimeActiveContext::try_from(&snapshot).expect("test runtime context")
    }

    fn cache_report(actual_cached_tokens: Option<u64>) -> CacheUsageReport {
        CacheUsageReport {
            configured: true,
            hint_serialized: true,
            retention_sent: None,
            stable_prefix_segments: 2,
            stable_prompt_tokens: 400,
            volatile_prompt_tokens: 60,
            cacheable_prefix_tokens: 350,
            stable_after_boundary_tokens: 50,
            local_prefix_fingerprint: Some("prefix-a".into()),
            routing_key: Some("route-a".into()),
            actual_cached_tokens,
        }
    }

    fn sample_context_state() -> crate::tui::state::ContextPaneState {
        let tree = ContextTreeState::replay(&[ContextTreeOp::CreateNode {
            node_id: ContextNodeId::new("node-1").expect("node id"),
            parent_node_id: Some(ContextNodeId::root()),
            label: Some("Active task".into()),
            purpose: Some("Track current work".into()),
            block_ref: None,
            source_ref: None,
        }])
        .expect("tree");
        let mut blocks = BTreeMap::new();
        let block_id = ContextBlockId::new("block-1").expect("block id");
        blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: Some("node-1".into()),
                kind: ContextBlockKind::Note,
                title: "Current plan".into(),
                detail: "Outline next steps".into(),
                source: ContextBlockSource::TranscriptSpan {
                    start_sequence: 1,
                    end_sequence: 2,
                },
                source_start_sequence: Some(1),
                available_sequence: Some(2),
                protected_reasons: Vec::new(),
            },
        );

        crate::tui::state::ContextPaneState {
            tree,
            view: ContextViewProjection {
                blocks,
                ..ContextViewProjection::default()
            },
            runtime_context: None,
            open_detail: None,
        }
    }

    fn sample_question_request(multiple: bool) -> crate::tool::QuestionRequest {
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: if multiple {
                    "Choose several".into()
                } else {
                    "Choose one".into()
                },
                header: "Mode".into(),
                options: vec![
                    crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Fast path".into(),
                    },
                    crate::tool::QuestionOption {
                        label: "Safe".into(),
                        description: "Safe path".into(),
                    },
                ],
                multiple,
            }],
        }
    }

    fn sample_multi_question_request() -> crate::tool::QuestionRequest {
        crate::tool::QuestionRequest {
            questions: vec![
                crate::tool::QuestionSpec {
                    question: "Choose one".into(),
                    header: "Mode".into(),
                    options: vec![crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Fast path".into(),
                    }],
                    multiple: false,
                },
                crate::tool::QuestionSpec {
                    question: "Choose tone".into(),
                    header: "Tone".into(),
                    options: vec![crate::tool::QuestionOption {
                        label: "Warm".into(),
                        description: "Warm path".into(),
                    }],
                    multiple: false,
                },
            ],
        }
    }

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = mpsc::unbounded_channel();
        let base = std::env::temp_dir().join(format!(
            "letcode-tui-runtime-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![AvailableModel::with_context_window_and_reasoning(
                "gpt-5.5",
                "GPT-5.5",
                None,
                None,
                vec![
                    ModelReasoningEffort::None,
                    ModelReasoningEffort::Minimal,
                    ModelReasoningEffort::Low,
                    ModelReasoningEffort::Medium,
                    ModelReasoningEffort::High,
                    ModelReasoningEffort::Xhigh,
                ],
            )],
            std::env::temp_dir(),
            base,
        )
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(
            Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("https://api.openai.com/v1")
                    .with_api_key("test"),
            ),
            "gpt-test",
            4,
            4,
        )
    }

    #[test]
    fn submit_records_prompt_and_updates_state() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .show_toast("stale notice", ToastKind::Info);
        runtime.state_mut().set_input("hello world");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SubmitPrompt("hello world".into()))
        );
        assert!(runtime.state().input_buffer.is_empty());
        assert_eq!(runtime.state().phase, AppPhase::Running);
        assert!(runtime.state().active_session);
        assert_eq!(runtime.submitted_prompts(), &["hello world".to_string()]);
        assert!(runtime.state().timeline.items().is_empty());
        assert!(runtime.state().toast().is_none());
    }

    #[test]
    fn skill_marker_with_adjacent_cjk_submits_full_prompt() {
        let input = "@skill(humanizer-zh)这个skill是干什么的";
        let mut runtime = runtime();
        runtime.state_mut().set_input(input);

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, Some(RuntimeCommand::SubmitPrompt(input.into())));
        assert_eq!(runtime.submitted_prompts(), &[input.to_string()]);
    }

    #[test]
    fn mcp_command_opens_picker_with_server_status() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.state_mut().set_input("/mcp");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("command opens picker");

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_eq!(dialog.kind, DialogKind::McpPicker);
        assert_eq!(dialog.items[0].label, "docs");
        assert_eq!(
            dialog.items[0].right_detail.as_deref(),
            Some("● Online · 1 tools")
        );
    }

    #[test]
    fn mcp_discovered_tools_refresh_an_open_picker() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/mcp");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("command opens picker");
        runtime
            .state_mut()
            .dialog_mut()
            .expect("MCP picker")
            .insert_query_char('d');

        runtime.apply_runner_event(RunnerEvent::McpToolsDiscovered(vec![
            crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            },
        ]));

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_eq!(dialog.description, None);
        assert_eq!(dialog.query, "d");
        assert_eq!(dialog.items[0].label, "docs");
    }

    #[test]
    fn mcp_picker_redacts_discovery_error_when_opened_unavailable() {
        let mut runtime = runtime();
        let diagnostic = "failed to reach https://mcp.internal.example: connection refused";
        runtime
            .state_mut()
            .mark_mcp_discovery_unavailable(diagnostic.into());

        runtime.show_mcp_dialog().expect("opens picker");

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_eq!(dialog.kind, DialogKind::McpPicker);
        assert_eq!(
            dialog.description.as_deref(),
            Some(MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION)
        );
        assert_ne!(dialog.description.as_deref(), Some(diagnostic));
        assert!(
            !dialog
                .description
                .as_deref()
                .is_some_and(|text| text.contains(diagnostic))
        );
    }

    #[test]
    fn mcp_open_picker_refresh_redacts_discovery_error_and_preserves_state() {
        let mut runtime = runtime();
        let diagnostic = "failed to reach https://mcp.internal.example: connection refused";

        runtime.show_mcp_dialog().expect("opens picker");
        assert_eq!(
            runtime
                .state()
                .dialog()
                .expect("MCP picker")
                .description
                .as_deref(),
            Some(MCP_DISCOVERY_LOADING_DESCRIPTION)
        );

        runtime.apply_runner_event(RunnerEvent::McpDiscoveryUnavailable(diagnostic.into()));

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_eq!(
            dialog.description.as_deref(),
            Some(MCP_DISCOVERY_UNAVAILABLE_DESCRIPTION)
        );
        assert_ne!(dialog.description.as_deref(), Some(diagnostic));
        assert!(
            !dialog
                .description
                .as_deref()
                .is_some_and(|text| text.contains(diagnostic))
        );
    }

    #[test]
    fn mcp_diagnostic_is_visible_in_the_timeline_without_picker_detail() {
        let mut runtime = runtime();
        let message = "failed to discover MCP tools: connection refused";

        runtime.state_mut().set_input("/mcp");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("command opens picker");
        runtime.apply_runner_event(RunnerEvent::McpDiagnostic(message.into()));

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_ne!(dialog.description.as_deref(), Some(message));
        assert!(runtime.state().timeline.items().is_empty());
        assert!(matches!(
            runtime.state().toast(),
            Some(toast) if toast.message == message && toast.kind == ToastKind::Error
        ));
    }

    #[test]
    fn mcp_startup_offline_server_is_visible_in_picker_and_timeline() {
        let mut runtime = runtime();
        let message = "MCP server 'docs' is offline: connection refused";

        runtime.state_mut().set_input("/mcp");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("command opens picker");
        runtime.apply_runner_event(RunnerEvent::McpDiagnostic(message.into()));
        runtime.apply_runner_event(RunnerEvent::McpToolsDiscovered(vec![
            crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Offline {
                    message: "connection refused".into(),
                },
            },
        ]));

        let dialog = runtime.state().dialog().expect("MCP picker");
        assert_eq!(dialog.description, None);
        assert_eq!(dialog.items[0].label, "docs");
        assert_eq!(dialog.items[0].right_detail.as_deref(), Some("● Offline"));
        assert!(
            !runtime
                .state()
                .timeline
                .items()
                .iter()
                .any(|item| matches!(item, TimelineItem::Error(_)))
        );
        assert!(matches!(
            runtime.state().toast(),
            Some(toast) if toast.message == message && toast.kind == ToastKind::Error
        ));
    }

    #[test]
    fn mcp_toggle_is_rejected_while_a_turn_is_running() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.show_mcp_dialog().expect("opens picker");
        runtime.runner_turn_active = true;

        let command = runtime
            .handle_input_action(InputAction::DialogToggle)
            .expect("toggle is rejected");

        assert_eq!(command, None);
        assert!(!runtime.state().mcp_updating.contains("docs"));
    }

    #[test]
    fn mcp_enter_opens_tools_and_escape_returns_to_servers() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.state_mut().set_mcp_server_tools(
            "docs".into(),
            vec![crate::mcp::McpToolCatalogEntry {
                name: "lookup-docs".into(),
                description: "Find documentation".into(),
            }],
        );
        runtime.show_mcp_dialog().expect("opens servers");

        assert_eq!(runtime.handle_dialog_accept().expect("opens tools"), None);
        let dialog = runtime.state().dialog().expect("tools picker");
        assert_eq!(dialog.kind, DialogKind::McpToolsPicker);
        assert_eq!(dialog.items[0].label, "lookup-docs");
        assert_eq!(
            dialog.items[0].detail.as_deref(),
            Some("Find documentation")
        );

        runtime
            .handle_input_action(InputAction::DialogCancel)
            .expect("returns to servers");
        assert_eq!(
            runtime.state().dialog().expect("server picker").kind,
            DialogKind::McpPicker
        );
    }

    #[test]
    fn mcp_tools_picker_refreshes_for_tools_then_server_updates() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.state_mut().set_mcp_server_tools(
            "docs".into(),
            vec![crate::mcp::McpToolCatalogEntry {
                name: "lookup".into(),
                description: "Find documentation".into(),
            }],
        );
        runtime.show_mcp_dialog().expect("opens servers");
        runtime.handle_dialog_accept().expect("opens tools");
        runtime
            .state_mut()
            .dialog_mut()
            .expect("tools picker")
            .query = "look".into();

        runtime.apply_runner_event(RunnerEvent::McpServerToolsUpdated {
            name: "docs".into(),
            tools: vec![
                crate::mcp::McpToolCatalogEntry {
                    name: "lookup".into(),
                    description: "Find documentation".into(),
                },
                crate::mcp::McpToolCatalogEntry {
                    name: "search".into(),
                    description: "Search documentation".into(),
                },
            ],
        });

        let dialog = runtime.state().dialog().expect("tools picker remains open");
        assert_eq!(dialog.kind, DialogKind::McpToolsPicker);
        assert_eq!(dialog.mcp_server_name.as_deref(), Some("docs"));
        assert_eq!(dialog.query, "look");
        assert_eq!(
            dialog.selected_item().map(|item| item.id.as_str()),
            Some("lookup")
        );
        assert_eq!(dialog.items.len(), 2);
        assert_eq!(
            dialog.description.as_deref(),
            Some("Online · 1 tools available")
        );

        runtime.apply_runner_event(RunnerEvent::McpServerUpdated(
            crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 2 },
            },
        ));

        let dialog = runtime.state().dialog().expect("tools picker remains open");
        assert_eq!(dialog.kind, DialogKind::McpToolsPicker);
        assert_eq!(dialog.query, "look");
        assert_eq!(
            dialog.selected_item().map(|item| item.id.as_str()),
            Some("lookup")
        );
        assert_eq!(
            dialog.description.as_deref(),
            Some("Online · 2 tools available")
        );
    }

    #[test]
    fn mcp_tools_back_restores_primary_picker_query_and_selection() {
        let mut runtime = runtime();
        runtime.state_mut().set_mcp_servers(vec![
            crate::mcp::McpServerCatalogEntry {
                name: "alpha".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            },
            crate::mcp::McpServerCatalogEntry {
                name: "beta".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            },
        ]);
        runtime.show_mcp_dialog().expect("opens servers");
        let dialog = runtime.state_mut().dialog_mut().expect("server picker");
        dialog.query = "bet".into();
        dialog.selected = 1;
        runtime.handle_dialog_accept().expect("opens tools");

        runtime.apply_runner_event(RunnerEvent::McpServerUpdated(
            crate::mcp::McpServerCatalogEntry {
                name: "beta".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Offline {
                    message: "connection refused".into(),
                },
            },
        ));
        runtime
            .handle_input_action(InputAction::DialogCancel)
            .expect("returns to servers");

        let dialog = runtime.state().dialog().expect("server picker");
        assert_eq!(dialog.kind, DialogKind::McpPicker);
        assert_eq!(dialog.query, "bet");
        assert_eq!(
            dialog.selected_item().map(|item| item.id.as_str()),
            Some("beta")
        );
        assert_eq!(dialog.items[1].right_detail.as_deref(), Some("● Offline"));
    }

    #[test]
    fn mcp_space_requests_a_toggle_without_opening_tools() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.show_mcp_dialog().expect("opens servers");

        let command = runtime
            .handle_input_action(InputAction::DialogToggle)
            .expect("requests toggle");

        assert_eq!(
            command,
            Some(RuntimeCommand::ToggleMcpServer("docs".into()))
        );
        assert!(runtime.state().mcp_updating.contains("docs"));
        assert_eq!(
            runtime.state().dialog().expect("server picker").kind,
            DialogKind::McpPicker
        );
    }

    #[test]
    fn mcp_toggle_is_rejected_while_the_server_is_updating() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 1 },
            }]);
        runtime.show_mcp_dialog().expect("opens picker");

        let first_command = runtime
            .handle_input_action(InputAction::DialogToggle)
            .expect("toggle starts");
        let second_command = runtime
            .handle_input_action(InputAction::DialogToggle)
            .expect("duplicate toggle is rejected");

        assert_eq!(
            first_command,
            Some(RuntimeCommand::ToggleMcpServer("docs".into()))
        );
        assert_eq!(second_command, None);
        assert!(runtime.state().mcp_updating.contains("docs"));
        assert!(matches!(
            runtime.state().dialog(),
            Some(dialog) if dialog.kind == DialogKind::McpPicker
        ));
    }

    #[test]
    fn skill_command_opens_empty_picker() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/skill");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("command opens picker");

        let dialog = runtime.state().dialog().expect("skill picker");
        assert_eq!(dialog.kind, DialogKind::SkillPicker);
        assert!(dialog.items.is_empty());
    }

    #[test]
    fn skill_picker_attaches_a_deduplicated_token_to_the_draft() {
        let mut runtime = runtime();
        runtime.state_mut().set_skill_cards(vec![SkillCard {
            name: "rust-audit".into(),
            description: "Review Rust code".into(),
            location: ".agents/skills".into(),
            path: PathBuf::from("/repo/.agents/skills/rust-audit/SKILL.md"),
        }]);
        runtime.state_mut().set_input("Review this module");
        runtime.show_skill_dialog().expect("opens picker");

        runtime.handle_dialog_accept().expect("attaches skill");

        assert_eq!(
            runtime.state().input_buffer,
            format!(
                "Review this module{}",
                crate::tui::state::COMPOSER_ATTACHMENT_MARKER
            )
        );
        assert_eq!(
            runtime.state().composer_tokens[0].skill_name(),
            Some("rust-audit")
        );
        assert!(!runtime.state().dialog_is_open());

        runtime.show_skill_dialog().expect("opens picker");
        runtime.handle_dialog_accept().expect("deduplicates skill");
        assert_eq!(runtime.state().composer_tokens.len(), 1);
        let content = runtime.state().composer_content();
        assert_eq!(content.text, "Review this module");
        assert_eq!(content.selected_skills, vec!["rust-audit"]);
    }

    #[test]
    fn context_detail_dialog_is_distinct_from_context_picker() {
        let detail = context_detail_dialog(
            &sample_context_state(),
            &ContextDetailTarget::Block("block-1".into()),
        )
        .expect("detail dialog");

        assert_eq!(detail.kind, DialogKind::ContextDetail);
        assert!(detail.title.starts_with("Detail · "));
        assert!(detail.description.is_none());
    }

    #[test]
    fn invalid_context_selection_surfaces_clear_feedback() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::ContextPicker,
            "Context",
            None,
            vec![DialogItem::new("broken-id", "Broken", None)],
        ));

        runtime.sync_context_inspector_preview();

        assert!(runtime.state().toast().is_some());
        assert!(runtime.state().timeline.items().is_empty());
    }

    #[test]
    fn invalid_context_selection_in_child_view_uses_active_timeline_feedback() {
        let mut runtime = runtime();
        let records = vec![TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        }];
        runtime.state_mut().replace_child_timeline_from_records(
            &records,
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        runtime
            .state_mut()
            .set_child_context_for_test(sample_context_state());
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::ContextPicker,
            "Context",
            None,
            vec![DialogItem::new("broken-id", "Broken", None)],
        ));

        runtime.sync_context_inspector_preview();

        assert!(runtime.state().active_timeline().items().is_empty());
        assert!(runtime.state().timeline.items().is_empty());
    }

    #[test]
    fn unavailable_context_detail_surfaces_clear_feedback() {
        let mut runtime = runtime();
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::ContextPicker,
            "Context",
            None,
            vec![DialogItem::new("block:block-1", "Missing block", None)],
        ));

        runtime.sync_context_inspector_preview();

        assert!(runtime.state().active_context().open_detail.is_none());
        assert!(runtime.state().toast().is_some());
    }

    #[test]
    fn context_picker_enter_keeps_inspector_open() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.show_context_dialog().expect("show context dialog");

        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("accept handled");

        assert!(matches!(
            runtime.state().dialog().map(|dialog| &dialog.kind),
            Some(DialogKind::ContextPicker)
        ));
        assert!(runtime.state().active_context().open_detail.is_some());
        assert!(
            runtime
                .state()
                .dialog()
                .is_some_and(|dialog| dialog.detail_focused)
        );
    }

    #[test]
    fn context_picker_esc_in_detail_mode_returns_to_list_mode() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.show_context_dialog().expect("show context dialog");
        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("accept handled");

        runtime
            .handle_input_action(InputAction::DialogCancel)
            .expect("cancel handled");

        assert!(matches!(
            runtime.state().dialog().map(|dialog| &dialog.kind),
            Some(DialogKind::ContextPicker)
        ));
        assert!(
            runtime
                .state()
                .dialog()
                .is_some_and(|dialog| !dialog.detail_focused)
        );
    }

    #[test]
    fn context_picker_detail_mode_scrolls_without_moving_selection() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.show_context_dialog().expect("show context dialog");
        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("accept handled");

        let selected_before = runtime.state().dialog().map(|dialog| dialog.selected);
        runtime
            .handle_input_action(InputAction::DialogNext)
            .expect("scroll handled");

        let dialog = runtime.state().dialog().expect("dialog open");
        assert_eq!(Some(dialog.selected), selected_before);
        assert!(dialog.detail_scroll > 0);
    }

    #[test]
    fn context_picker_detail_scroll_stops_at_max() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.show_context_dialog().expect("show context dialog");
        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("detail focus handled");
        if let Some(dialog) = runtime.state_mut().dialog.as_mut() {
            dialog.detail_scroll_max = 1;
        }

        runtime
            .handle_input_action(InputAction::DialogNext)
            .expect("scroll handled");
        runtime
            .handle_input_action(InputAction::DialogNext)
            .expect("scroll handled");

        let dialog = runtime.state().dialog().expect("dialog open");
        assert_eq!(dialog.detail_scroll, 1);
    }

    #[test]
    fn context_picker_enter_in_detail_mode_does_not_reset_scroll() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.show_context_dialog().expect("show context dialog");
        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("detail focus handled");
        runtime
            .handle_input_action(InputAction::DialogNext)
            .expect("scroll handled");
        let scroll_before = runtime
            .state()
            .dialog()
            .map(|dialog| dialog.detail_scroll)
            .unwrap_or_default();

        runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("accept handled");

        let dialog = runtime.state().dialog().expect("dialog open");
        assert!(dialog.detail_focused);
        assert_eq!(dialog.detail_scroll, scroll_before);
    }

    #[test]
    fn context_picker_opens_on_current_detail_target() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime
            .state_mut()
            .open_context_detail(Some(ContextDetailTarget::Block("block-1".into())));

        runtime.show_context_dialog().expect("show context dialog");

        let selected_id = runtime
            .state()
            .dialog()
            .and_then(|dialog| dialog.selected_item())
            .map(|item| item.id.as_str());
        assert_eq!(selected_id, Some("block:block-1"));
    }

    #[tokio::test]
    async fn single_question_pick_option_submits_immediately() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        runtime
            .state_mut()
            .show_toast("stale notice", ToastKind::Info);

        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime
            .handle_input_action(InputAction::QuestionPickOption(1))
            .expect("pick succeeds");

        assert_eq!(
            rx.await.expect("answer received"),
            Ok(crate::tool::QuestionResponse {
                answers: vec![vec!["Fast".into()]],
            })
        );
        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.state().toast().is_none());
    }

    #[test]
    fn submit_question_with_dropped_receiver_clears_ui_without_error() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        drop(rx);
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        assert!(
            runtime
                .handle_input_action(InputAction::QuestionPickOption(1))
                .is_ok()
        );
        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
    }

    #[test]
    fn cancel_question_with_dropped_receiver_clears_ui_without_error() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        drop(rx);
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        assert!(
            runtime
                .handle_input_action(InputAction::QuestionCancel)
                .is_ok()
        );
        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
    }

    #[tokio::test]
    async fn cancel_question_delivers_cancellation() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime
            .handle_input_action(InputAction::QuestionCancel)
            .expect("cancel succeeds");

        assert_eq!(
            rx.await.expect("cancellation received"),
            Err("question dismissed by user".into())
        );
        assert!(runtime.state().pending_question.is_none());
    }

    #[test]
    fn submit_without_question_is_a_diagnostic_noop() {
        let mut runtime = runtime();

        runtime
            .handle_input_action(InputAction::QuestionSubmit)
            .expect("submit is ignored");
    }

    #[test]
    fn multi_select_enter_toggles_without_submitting() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();

        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(true),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime
            .handle_input_action(InputAction::QuestionActivate)
            .expect("toggle succeeds");

        let question = runtime
            .state()
            .pending_question
            .as_ref()
            .expect("question still pending");
        assert_eq!(question.questions[0].answers(), vec!["Fast".to_string()]);
    }

    #[test]
    fn confirm_submit_focuses_first_unanswered_question() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();

        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_multi_question_request(),
            handle: RunnerQuestionRequest::new(tx),
        });

        {
            let question = runtime
                .state_mut()
                .pending_question
                .as_mut()
                .expect("question pending");
            question.questions[0].selected_labels.push("Fast".into());
            question.active_tab = 2;
        }

        runtime
            .handle_input_action(InputAction::QuestionSubmit)
            .expect("submit handled");

        let question = runtime
            .state()
            .pending_question
            .as_ref()
            .expect("question still pending");
        assert_eq!(question.active_tab, 1);
    }

    #[test]
    fn esc_in_custom_edit_only_closes_editor() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();

        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        {
            let question = runtime
                .state_mut()
                .pending_question
                .as_mut()
                .expect("question pending");
            question.active_row = 2;
            question.begin_custom_edit();
        }

        runtime
            .handle_input_action(InputAction::QuestionCancel)
            .expect("cancel handled");

        let question = runtime
            .state()
            .pending_question
            .as_ref()
            .expect("question still pending");
        assert!(!question.editing_custom);
    }

    #[test]
    fn history_tree_selection_restores_user_content_and_targets_the_parent() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-history-selection-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&sessions_dir).expect("sessions directory");
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("recorder");
        let session_id = recorder.session_id().to_string();
        recorder.record_user_message("first").expect("first user");
        recorder
            .record_assistant_message("first answer")
            .expect("first answer");
        recorder.record_user_message("second").expect("second user");
        drop(recorder);
        let transcript = Arc::new(StdMutex::new(
            TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("open recorder"),
        ));
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        let (initial_session_id, _) = initial_session_metadata(&transcript).expect("load metadata");
        state.session_id = Some(initial_session_id);
        assert_eq!(state.session_id.as_deref(), Some(session_id.as_str()));
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime =
            TuiRuntime::new(state, rx, Vec::new(), sessions_dir, std::env::temp_dir());

        let path = runtime.sessions_dir.join(format!("{session_id}.jsonl"));
        let records = read_records(&path).expect("records");
        runtime.apply_runner_event(RunnerEvent::SessionHistoryLoaded {
            entries: transcript_projection::project_session_history_tree(&records),
        });
        let dialog = runtime.state().dialog().expect("history dialog");
        assert_eq!(dialog.kind, DialogKind::HistoryTree);
        assert_eq!(dialog.selected_item().expect("selected").id, "entry-3");

        let command = runtime
            .handle_dialog_accept()
            .expect("accept user selection");
        assert_eq!(
            command,
            Some(RuntimeCommand::NavigateHistory {
                target_entry_id: "entry-2".into(),
            })
        );
        assert_eq!(runtime.state().input_buffer, "second");

        runtime.apply_runner_event(RunnerEvent::SessionHistoryLoaded {
            entries: transcript_projection::project_session_history_tree(&records),
        });
        runtime
            .state_mut()
            .dialog_mut()
            .expect("history dialog")
            .selected = 1;
        let command = runtime
            .handle_dialog_accept()
            .expect("accept assistant selection");
        assert_eq!(
            command,
            Some(RuntimeCommand::NavigateHistory {
                target_entry_id: "entry-2".into(),
            })
        );
    }

    #[test]
    fn history_tree_acceptance_rechecks_pending_running_queued_question_and_inflight_state() {
        for state in ["running", "permission", "queued", "question", "inflight"] {
            let mut runtime = runtime();
            runtime.state_mut().open_dialog(DialogState::new(
                DialogKind::HistoryTree,
                "Session history",
                None,
                vec![DialogItem::new("entry-1", "You: first", None)],
            ));
            match state {
                "running" => runtime.runner_turn_active = true,
                "permission" => {
                    let (reply_tx, _reply_rx) = oneshot::channel();
                    runtime.apply_runner_event(RunnerEvent::PermissionRequested {
                        event: PermissionRequestEvent::new("call-1", "shell__exec", "Run command"),
                        handle: RunnerPermissionRequest::new(reply_tx),
                    });
                }
                "queued" => runtime
                    .queued_prompts
                    .push_back(UserMessageSubmission::from("queued")),
                "question" => {
                    let (reply_tx, _reply_rx) = oneshot::channel();
                    runtime.apply_runner_event(RunnerEvent::QuestionRequested {
                        request: sample_question_request(false),
                        handle: RunnerQuestionRequest::new(reply_tx),
                    });
                }
                "inflight" => runtime
                    .queued_prompt_lifecycle
                    .dispatch(UserMessageSubmission::from("inflight")),
                _ => unreachable!(),
            }

            assert_eq!(
                runtime
                    .handle_dialog_accept()
                    .expect("accept history dialog"),
                None,
                "{state} state must reject navigation"
            );
            assert!(!runtime.state().dialog_is_open());
        }
    }

    #[test]
    fn navigation_commands_are_blocked_for_running_pending_and_queued_turns() {
        for state in ["running", "pending", "queued"] {
            let mut runtime = runtime();
            match state {
                "running" => runtime.runner_turn_active = true,
                "pending" => {
                    let (reply_tx, _reply_rx) = oneshot::channel();
                    runtime.apply_runner_event(RunnerEvent::PermissionRequested {
                        event: PermissionRequestEvent::new("call-1", "shell__exec", "Run command"),
                        handle: RunnerPermissionRequest::new(reply_tx),
                    });
                }
                "queued" => runtime
                    .queued_prompt_lifecycle
                    .dispatch(UserMessageSubmission::from("queued")),
                _ => unreachable!(),
            }
            runtime.state_mut().set_input("/undo");
            assert_eq!(
                runtime
                    .handle_input_action(InputAction::Submit)
                    .expect("submit"),
                None,
                "{state} turn must block undo"
            );
            assert_eq!(runtime.state().input_buffer, "/undo");
        }
    }

    #[test]
    fn history_navigation_preserves_draft_and_restores_it() {
        let mut runtime = runtime();
        runtime.submitted_prompts = vec!["first".into(), "second".into()];
        runtime.state_mut().set_input("draft");

        runtime
            .handle_input_action(InputAction::HistoryPrev)
            .expect("history prev succeeds");
        assert_eq!(runtime.state().input_buffer, "second");

        runtime
            .handle_input_action(InputAction::HistoryPrev)
            .expect("history prev succeeds");
        assert_eq!(runtime.state().input_buffer, "first");

        runtime
            .handle_input_action(InputAction::HistoryNext)
            .expect("history next succeeds");
        assert_eq!(runtime.state().input_buffer, "second");

        runtime
            .handle_input_action(InputAction::HistoryNext)
            .expect("history next restores draft");
        assert_eq!(runtime.state().input_buffer, "draft");
    }

    #[test]
    fn history_navigation_restores_attachment_bearing_draft() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });
        runtime.submitted_prompts = vec!["previous".into()];

        runtime
            .handle_input_action(InputAction::HistoryPrev)
            .expect("history prev succeeds");

        assert_eq!(runtime.state().input_buffer, "previous");
        assert!(runtime.state().composer_tokens.is_empty());

        runtime
            .handle_input_action(InputAction::HistoryNext)
            .expect("history next restores draft");
        assert_eq!(
            runtime.state().input_buffer,
            crate::tui::state::COMPOSER_ATTACHMENT_MARKER_STR
        );
        assert_eq!(
            runtime.state().input_cursor,
            crate::tui::state::COMPOSER_ATTACHMENT_MARKER.len_utf8()
        );
        assert_eq!(
            runtime.state().composer_tokens[0].image().unwrap().id,
            "img-1"
        );
    }

    #[test]
    fn commands_with_attachments_leave_the_composer_draft_intact() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/help");
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });
        let before = runtime.state().input_buffer.clone();

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command handling succeeds");

        assert_eq!(command, None);
        assert_eq!(runtime.state().input_buffer, before);
        assert_eq!(
            runtime.state().composer_tokens[0].image().unwrap().id,
            "img-1"
        );
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Remove attachments before running command")
        );
    }

    #[test]
    fn compact_with_attachments_has_no_runtime_side_effects() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/compact");
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });
        let draft = runtime.state().input_buffer.clone();
        let phase = runtime.state().phase;

        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("command handling succeeds"),
            None
        );
        assert_eq!(runtime.state().phase, phase);
        assert!(!runtime.runner_turn_active);
        assert_eq!(runtime.state().input_buffer, draft);
        assert_eq!(
            runtime.state().composer_tokens[0].image().unwrap().id,
            "img-1"
        );
    }

    #[test]
    fn rejected_attachment_command_preserves_history_draft_navigation() {
        let mut runtime = runtime();
        runtime.submitted_prompts = vec!["previous".into()];
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "draft-image".into(),
                label: "draft.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });

        runtime
            .handle_input_action(InputAction::HistoryPrev)
            .expect("history prev succeeds");
        runtime.state_mut().set_input("/compact");
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "command-image".into(),
                label: "command.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,BBBB".into(),
            });

        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("command rejection succeeds"),
            None
        );
        runtime
            .handle_input_action(InputAction::HistoryNext)
            .expect("history next restores draft");

        assert_eq!(
            runtime.state().input_buffer,
            crate::tui::state::COMPOSER_ATTACHMENT_MARKER_STR
        );
        assert_eq!(
            runtime.state().composer_tokens[0].image().unwrap().id,
            "draft-image"
        );
    }

    #[test]
    fn slash_selection_with_attachments_does_not_replace_the_draft() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/m");
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });
        let draft = runtime.state().input_buffer.clone();

        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("slash selection handling succeeds"),
            None
        );
        assert_eq!(runtime.state().input_buffer, draft);
        assert_eq!(
            runtime.state().composer_tokens[0].image().unwrap().id,
            "img-1"
        );
        assert!(runtime.state().dialog().is_none());
    }

    #[test]
    fn command_parsing_ignores_composer_token_markers() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/help");
        assert!(runtime.state_mut().add_composer_skill("rust-audit".into()));

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command handling succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Remove attachments before running command")
        );
    }

    #[test]
    fn attachment_only_submit_is_treated_as_a_prompt() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-only".into(),
                label: "clipboard".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("attachment-only submit succeeds");
        let Some(RuntimeCommand::SubmitPrompt(submission)) = command else {
            panic!("expected prompt submission");
        };

        assert_eq!(submission.content.text, "[Image 1]");
        assert_eq!(submission.content.attachments[0].id, "img-only");
    }

    #[test]
    fn submit_preserves_inline_part_order() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("before after");
        runtime.state_mut().input_cursor = "before ".len();
        runtime
            .state_mut()
            .add_composer_attachment(UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            });

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");
        let Some(RuntimeCommand::SubmitPrompt(submission)) = command else {
            panic!("expected prompt submission");
        };

        assert_eq!(submission.content.text, "before [Image 1]after");
        assert_eq!(
            submission.content.parts(),
            vec![
                crate::user_content::UserMessagePart::Text {
                    text: "before ".into(),
                },
                crate::user_content::UserMessagePart::Text {
                    text: "[Image 1]".into(),
                },
                crate::user_content::UserMessagePart::Image {
                    attachment: UserImageAttachment {
                        id: "img-1".into(),
                        label: "screen.png".into(),
                        mime: "image/png".into(),
                        data_url: "data:image/png;base64,AAAA".into(),
                    },
                },
                crate::user_content::UserMessagePart::Text {
                    text: "after".into(),
                },
            ]
        );
    }

    #[test]
    fn editing_resets_history_navigation_to_current_buffer() {
        let mut runtime = runtime();
        runtime.submitted_prompts = vec!["first".into(), "second".into()];

        runtime
            .handle_input_action(InputAction::HistoryPrev)
            .expect("history prev succeeds");
        assert_eq!(runtime.state().input_buffer, "second");

        runtime
            .handle_input_action(InputAction::Insert('!'))
            .expect("insert succeeds");
        assert_eq!(runtime.state().input_buffer, "second!");

        runtime
            .handle_input_action(InputAction::HistoryNext)
            .expect("history next is cleared");
        assert_eq!(runtime.state().input_buffer, "second!");
    }

    #[test]
    fn scroll_actions_update_bottom_relative_offset() {
        let mut runtime = runtime();

        runtime
            .handle_input_action(InputAction::ScrollUp)
            .expect("scroll up succeeds");
        assert_eq!(runtime.state().transcript_scroll_offset(), 1);
        assert!(!runtime.state().auto_scroll);

        runtime
            .handle_input_action(InputAction::ScrollPageUp)
            .expect("page up succeeds");
        assert_eq!(runtime.state().transcript_scroll_offset(), 11);

        runtime
            .handle_input_action(InputAction::ScrollDown)
            .expect("scroll down succeeds");
        assert_eq!(runtime.state().transcript_scroll_offset(), 10);

        runtime
            .handle_input_action(InputAction::ScrollPageDown)
            .expect("page down succeeds");
        assert_eq!(runtime.state().transcript_scroll_offset(), 0);
        assert!(runtime.state().auto_scroll);
    }

    #[test]
    fn paste_action_inserts_full_string_into_composer() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("helo");
        runtime.state_mut().input_cursor = 2;

        runtime
            .handle_input_action(InputAction::Paste("ll\nworld".into()))
            .expect("paste succeeds");

        assert_eq!(runtime.state().input_buffer, "hell\nworldlo");
        assert_eq!(runtime.state().input_cursor, 2 + "ll\nworld".len());
    }

    #[test]
    fn submit_then_runner_user_event_adds_single_user_message() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("hello world");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");
        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new(
            "hello world",
        )));

        assert_eq!(runtime.state().timeline.items().len(), 1);
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(crate::tui::TimelineItem::User(message)) if message.text == "hello world"
        ));
    }

    #[test]
    fn matching_child_stream_event_updates_child_view_without_touching_parent() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "child-session".into(),
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
        });

        assert!(runtime.state().timeline.items().is_empty());
        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Assistant(message)) if message.text == "hello"
        ));
    }

    #[test]
    fn non_matching_child_stream_event_does_not_mutate_current_view() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "other-child".into(),
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new("hello")),
        });

        assert!(runtime.state().timeline.items().is_empty());
        assert!(runtime.state().active_timeline().items().is_empty());
    }

    #[test]
    fn child_interrupted_event_updates_child_view_without_touching_parent() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "child-session".into(),
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::Interrupted,
        });

        assert!(runtime.state().timeline.items().is_empty());
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Interrupted by user")
        );
    }

    #[test]
    fn slash_exit_and_plain_exit_quit_without_prompt_submission() {
        for command_text in ["/exit", "/quit", "exit", "quit"] {
            let mut runtime = runtime();
            runtime.state_mut().set_input(command_text);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("command succeeds");

            assert_eq!(command, None);
            assert!(runtime.state().quit_requested, "{command_text}");
            assert!(runtime.state().input_buffer.is_empty());
            assert!(runtime.submitted_prompts().is_empty());
        }
    }

    #[test]
    fn running_turn_blocks_exit_and_quit_commands() {
        for command_text in ["exit", "quit", "/exit", "/quit"] {
            let mut runtime = runtime();
            runtime.state_mut().phase = AppPhase::Running;
            runtime.state_mut().set_input(command_text);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("command is blocked while running");

            assert_eq!(command, None, "{command_text}");
            assert!(!runtime.state().quit_requested, "{command_text}");
            assert_eq!(runtime.queued_prompts.len(), 0, "{command_text}");
            assert!(runtime.submitted_prompts().is_empty(), "{command_text}");
            assert_eq!(runtime.state().input_buffer, command_text, "{command_text}");
        }
    }

    #[test]
    fn double_escape_confirms_running_turn_interrupt() {
        let mut runtime = runtime();
        runtime.state.phase = AppPhase::Running;

        let first = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first interrupt hint succeeds");
        assert_eq!(first, None);

        let second = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("second interrupt returns command");
        assert_eq!(second, Some(RuntimeCommand::Interrupt));
    }

    #[test]
    fn interrupt_confirmation_survives_tick() {
        let mut runtime = runtime();
        runtime.state.phase = AppPhase::Running;

        runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first interrupt hint succeeds");
        runtime
            .handle_input_action(InputAction::Tick)
            .expect("tick succeeds");

        let second = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("second interrupt still confirms");
        assert_eq!(second, Some(RuntimeCommand::Interrupt));
    }

    #[test]
    fn interrupt_uses_runner_active_after_non_terminal_error() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state.phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new(
            "failed to view child transcript",
        )));
        assert_eq!(runtime.state().phase, AppPhase::Error);

        let first = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first interrupt hint succeeds");
        assert_eq!(first, None);

        let second = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("second interrupt returns command");
        assert_eq!(second, Some(RuntimeCommand::Interrupt));
    }

    #[test]
    fn interrupted_runner_event_returns_to_prompt_ready_state() {
        let mut runtime = runtime();
        runtime.state.phase = AppPhase::Running;

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert_eq!(runtime.state().phase, AppPhase::Completed);

        assert!(runtime.state().pending_permission.is_none());
    }

    #[tokio::test]
    async fn interrupted_cancels_parent_question_and_clears_local_state() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
        assert!(runtime.pending_question_child_session_id.is_none());
        assert_eq!(
            rx.await.expect("cancellation received"),
            Err("question cancelled because the turn was interrupted".into())
        );
    }

    #[test]
    fn interrupted_clears_question_when_receiver_was_dropped() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        drop(rx);
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
        assert!(runtime.pending_question_child_session_id.is_none());
    }

    #[tokio::test]
    async fn interrupted_cancels_child_question_and_clears_local_state() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::ChildQuestionRequested {
            child_session_id: "child-1".into(),
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
        assert!(runtime.pending_question_child_session_id.is_none());
        assert_eq!(
            rx.await.expect("cancellation received"),
            Err("question cancelled because the turn was interrupted".into())
        );
    }

    #[test]
    fn error_preserves_question_while_done_clears_parent_question() {
        let mut runtime = runtime();
        let (tx, mut rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::QuestionRequested {
            request: sample_question_request(false),
            handle: RunnerQuestionRequest::new(tx),
        });

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new("turn failed")));

        assert!(runtime.state().pending_question.is_some());
        assert!(matches!(
            rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        runtime.apply_runner_event(RunnerEvent::Done);

        assert!(runtime.state().pending_question.is_none());
        assert!(runtime.pending_question_handle.is_none());
        assert!(matches!(
            rx.try_recv(),
            Ok(Err(reason)) if reason == "question cancelled because the turn ended"
        ));
    }

    #[test]
    fn interrupt_rehydrates_agent_from_transcript() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-tui-runtime-interrupt-rehydrate-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        let transcript = Arc::new(StdMutex::new(recorder));

        {
            let mut recorder = transcript.lock().expect("lock recorder");
            recorder
                .record_user_message("unfinished")
                .expect("record user message");
            recorder
                .record_turn_interrupted(Some(1))
                .expect("record turn interrupted");
        }

        let mut agent = test_agent();
        agent
            .restore_session_history(vec![HistoryItem::user("stale dangling")], Vec::new(), 0)
            .expect("seed stale history");

        rehydrate_agent_from_transcript(&mut agent, &transcript).expect("rehydrate agent");

        assert!(matches!(
            agent.history_for_test(),
            [HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: assistant_text }]
                if content.text == "unfinished" && assistant_text.is_empty()
        ));
    }

    #[test]
    fn resumed_session_usage_uses_target_agent_and_drops_response_accounting() {
        let mut old_agent = test_agent();
        old_agent
            .restore_session_history(
                vec![HistoryItem::user("old session ".repeat(2_000))],
                Vec::new(),
                99,
            )
            .expect("restore old session");
        let old_usage = old_agent.session_token_usage().expect("old usage");

        let mut target_agent = test_agent();
        target_agent.set_model("target-model");
        target_agent
            .restore_session_history(vec![HistoryItem::user("target session")], Vec::new(), 2)
            .expect("restore target session");
        let target_frames = target_agent.protocol_frames_for_test().to_vec();
        let target_snapshot = target_agent.runtime_snapshot_for_test().clone();
        let expected_usage =
            restored_session_token_usage(&target_agent, target_agent.model(), &target_snapshot)
                .expect("target usage");
        let prepared_usage =
            restored_session_token_usage(&old_agent, target_agent.model(), &target_snapshot)
                .expect("prepare target usage");

        old_agent
            .restore_new_session_runtime_snapshot(target_frames, target_snapshot.clone(), 2)
            .expect("install target session");
        old_agent.set_model("target-model");
        let resumed_usage =
            restored_session_token_usage(&old_agent, old_agent.model(), &target_snapshot)
                .expect("resumed usage");

        assert_eq!(
            old_agent.runtime_snapshot_for_test().current_turn_id,
            Some(2)
        );
        assert_ne!(old_usage.used_tokens, expected_usage.used_tokens);
        assert_eq!(prepared_usage, expected_usage);
        assert_eq!(resumed_usage, expected_usage);
        assert_eq!(resumed_usage.output_tokens, 0);
        assert_eq!(resumed_usage.cached_tokens, 0);
        assert_eq!(resumed_usage.cache_report, None);
    }

    #[test]
    fn failed_candidate_usage_preserves_agent_and_recorder() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-runtime-candidate-usage-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
        let recorder_id = recorder.session_id().to_string();
        let recorder_path = recorder.path().to_path_buf();
        let mut agent = test_agent();
        agent
            .restore_session_history(vec![HistoryItem::user("old session")], Vec::new(), 7)
            .expect("restore old session");
        let model = agent.model().to_string();
        let history = agent.history_for_test().to_vec();
        let runtime_snapshot = agent.runtime_snapshot_for_test().clone();
        let mut invalid_metadata = crate::request_builder::ModelRequestMetadata::default();
        invalid_metadata.effective_input_limit_tokens = Some(0);
        agent.set_model_catalog(std::collections::HashMap::from([(
            String::from("invalid-model"),
            invalid_metadata,
        )]));
        let target_snapshot = test_agent().runtime_snapshot_for_test().clone();

        assert!(restored_session_token_usage(&agent, "invalid-model", &target_snapshot).is_err());

        assert_eq!(agent.model(), model);
        assert_eq!(agent.history_for_test(), history.as_slice());
        assert_eq!(agent.runtime_snapshot_for_test(), &runtime_snapshot);
        assert_eq!(recorder.session_id(), recorder_id);
        assert_eq!(recorder.path(), recorder_path);
    }

    #[test]
    fn permission_prompt_requires_double_esc_to_interrupt() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime
            .state_mut()
            .apply_event(AppEvent::PermissionRequested(
                crate::tui::events::PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
            ));

        let first = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first esc succeeds");
        assert_eq!(first, None);

        assert!(runtime.state().pending_permission.is_some());

        let second = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("second esc succeeds");
        assert_eq!(second, Some(RuntimeCommand::Interrupt));
    }

    #[test]
    fn slash_subagent_interrupt_terminalizes_parent_runtime_from_parent_view() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;

        send_subagent_interrupted(&tx, Some("child-session".into()));
        runtime.try_drain_runner_events();

        assert!(!runtime.runner_turn_active);
        assert_eq!(runtime.state().phase, AppPhase::Completed);
    }

    #[test]
    fn slash_subagent_interrupt_terminalizes_parent_runtime_from_child_view() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        send_subagent_interrupted(&tx, Some("child-session".into()));
        runtime.try_drain_runner_events();

        assert!(!runtime.runner_turn_active);
        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Interrupted by user")
        );
    }

    #[test]
    fn parent_interrupt_while_viewing_child_closes_child_active_tools() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        runtime.state_mut().apply_child_app_event(
            "child-session",
            AppEvent::ToolPending(crate::tui::events::ToolPendingEvent::new(
                "child-call",
                "fs__write",
            )),
        );

        let interrupt = runtime.build_interrupt_request();
        send_subagent_interrupted(&tx, interrupt.visible_child_session_id);
        runtime.try_drain_runner_events();

        assert!(!runtime.runner_turn_active);
        assert!(matches!(
            runtime.state().active_timeline().items().iter().find_map(|item| match item {
                crate::tui::TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            }),
            Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
        ));
    }

    #[test]
    fn interrupt_request_records_only_parent_tool_calls() {
        let mut runtime = runtime();
        runtime.state_mut().apply_event(AppEvent::ToolPending(
            crate::tui::events::ToolPendingEvent::new("parent-call", "shell__exec"),
        ));
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        runtime.state_mut().apply_child_app_event(
            "child-session",
            AppEvent::ToolPending(crate::tui::events::ToolPendingEvent::new(
                "child-call",
                "fs__write",
            )),
        );

        let interrupt = runtime.build_interrupt_request();

        assert_eq!(interrupt.parent_tool_calls.len(), 1);
        assert_eq!(interrupt.parent_tool_calls[0].0, "parent-call");
        assert_eq!(
            interrupt.visible_child_session_id.as_deref(),
            Some("child-session")
        );
    }

    #[test]
    fn child_transcript_view_blocks_parent_mutating_submit_paths() {
        for input in [
            "ask the parent agent",
            "@explorer inspect src/agent.rs",
            "@fixer wire agent__fixer tool",
            "@oracle review src/main.rs",
            "/new",
            "/resume abc123",
            "/model gpt-5.5-mini",
            "/permission safe",
        ] {
            let mut runtime = runtime();
            runtime.state_mut().replace_child_timeline_from_records(
                &[],
                "parent-session",
                "child-session",
                "explorer",
                0,
                1,
                1,
            );
            runtime.state_mut().set_input(input);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("submit succeeds");

            assert_eq!(command, None, "{input}");
            assert!(runtime.submitted_prompts().is_empty(), "{input}");
        }
    }

    #[test]
    fn child_transcript_view_allows_navigation_and_blocks_parent_shortcuts() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .replace_session_timeline_from_records(&[TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "parent prompt".into(),
                },
            }]);
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        let blocked = runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("shortcut succeeds");
        assert_eq!(blocked, None);

        runtime.state_mut().set_input("/child next");
        let child = runtime
            .handle_input_action(InputAction::Submit)
            .expect("child navigation succeeds");
        assert_eq!(
            child,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::Next,
                anchor_child_session_id: None
            })
        );

        runtime.state_mut().set_input("/parent");
        let parent = runtime
            .handle_input_action(InputAction::Submit)
            .expect("parent navigation succeeds");
        assert_eq!(parent, None);
        assert_eq!(
            runtime.state().transcript_view,
            crate::tui::state::TranscriptViewState::Parent
        );
        assert!(matches!(
            runtime.state().timeline.items().first(),
            Some(crate::tui::TimelineItem::User(message)) if message.text == "parent prompt"
        ));
    }

    #[test]
    fn running_turn_allows_child_navigation_commands() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("/child");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("child navigation succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::First,
                anchor_child_session_id: None
            })
        );
    }

    #[test]
    fn running_turn_opens_context_browser_locally() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_parent_context_for_test(sample_context_state());
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("/context");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("context browser opens while running");

        assert_eq!(command, None);
        assert!(runtime.state().input_buffer.is_empty());
        assert!(matches!(
            runtime.state().dialog().map(|dialog| &dialog.kind),
            Some(DialogKind::ContextPicker)
        ));
    }

    #[test]
    fn running_turn_handles_help_locally() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("/help");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("help succeeds while running");

        assert_eq!(command, None);
        assert!(runtime.state().input_buffer.is_empty());
        assert!(runtime.submitted_prompts().is_empty());
    }

    #[test]
    fn running_turn_queues_plain_prompts() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .show_toast("stale notice", ToastKind::Info);
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, None);
        assert!(runtime.state().input_buffer.is_empty());
        assert_eq!(runtime.state().phase, AppPhase::Running);
        assert_eq!(runtime.submitted_prompts(), &["follow up".to_string()]);
        assert_eq!(runtime.queued_prompts.len(), 1);
        assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));
        assert!(runtime.state().toast().is_none());
    }

    #[test]
    fn running_turn_preserves_selected_skills_in_queued_prompt() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        assert!(runtime.state_mut().add_composer_skill("rust-audit".into()));

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.queued_prompts[0].content.selected_skills,
            vec!["rust-audit"]
        );
        assert!(runtime.state().composer_tokens.is_empty());
    }

    #[test]
    fn running_turn_rejects_delegate_commands_without_queueing() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("@fixer fix failing test");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, None);
        assert_eq!(runtime.queued_prompts.len(), 0);
        assert!(runtime.submitted_prompts().is_empty());
        assert_eq!(runtime.state().input_buffer, "@fixer fix failing test");

        assert!(
            !runtime
                .state()
                .timeline
                .items()
                .iter()
                .any(|item| matches!(item, TimelineItem::Delegation(_)))
        );
    }

    #[test]
    fn expert_panel_accept_inserts_canonical_text() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@fi");

        runtime
            .handle_input_action(InputAction::SlashPanelAccept)
            .expect("accept succeeds");

        assert_eq!(runtime.state().input_buffer, "@fixer ");
    }

    #[test]
    fn queued_prompt_ack_requires_dispatched_prompt() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("same");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("same")));

        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["same".to_string()]
        );
        assert_eq!(
            runtime
                .state()
                .timeline
                .items()
                .iter()
                .filter(
                    |item| matches!(item, TimelineItem::User(message) if message.text == "same")
                )
                .count(),
            2
        );
        assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "same" && message.queued)));
    }

    #[test]
    fn queued_prompt_preview_does_not_reset_active_turn_state_until_ack() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().active_tool_call_id = Some("tool-1".into());
        runtime.state_mut().latest_todo = Some(crate::tui::timeline::TodoView {
            items: vec![TodoItem {
                id: "todo-1".into(),
                content: "keep working".into(),
                status: TodoStatus::InProgress,
            }],
            auto_continue: AutoContinueState {
                enabled: true,
                max_continuations: 2,
            },
        });
        runtime.state_mut().set_input("follow up");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        assert_eq!(
            runtime.state().active_tool_call_id.as_deref(),
            Some("tool-1")
        );
        assert!(runtime.state().latest_todo.is_some());

        runtime.apply_runner_event(RunnerEvent::Done);
        let Some(RuntimeCommand::SubmitPrompt(submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected queued submit command");
        };
        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::from_submission(
            submission,
        )));

        assert_eq!(runtime.state().active_tool_call_id, None);
        assert!(runtime.state().latest_todo.is_none());
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::User(message)) if message.text == "follow up" && !message.queued
        ));
    }

    #[test]
    fn delayed_mcp_diagnostic_after_done_preserves_completed_turn_todo_and_auto_continue() {
        let mut runtime = runtime();
        let message = "MCP server 'docs' is offline: connection refused";
        let auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 2,
        };
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().latest_auto_continue = auto_continue.clone();
        runtime.state_mut().latest_todo = Some(crate::tui::timeline::TodoView {
            items: vec![TodoItem {
                id: "todo-1".into(),
                content: "keep working".into(),
                status: TodoStatus::InProgress,
            }],
            auto_continue: auto_continue.clone(),
        });

        runtime.apply_runner_event(RunnerEvent::Done);
        runtime.apply_runner_event(RunnerEvent::McpDiagnostic(message.into()));

        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert_eq!(runtime.state().latest_auto_continue, auto_continue);
        assert_eq!(
            runtime
                .state()
                .latest_todo
                .as_ref()
                .expect("todo remains visible")
                .auto_continue,
            auto_continue
        );
        assert!(
            !runtime
                .state()
                .timeline
                .items()
                .iter()
                .any(|item| matches!(item, TimelineItem::Error(_)))
        );
        assert!(matches!(
            runtime.state().toast(),
            Some(toast) if toast.message == message && toast.kind == ToastKind::Error
        ));
    }

    #[test]
    fn dispatched_queued_prompt_failure_before_ack_clears_handoff_without_redispatch() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);
        let Some(RuntimeCommand::SubmitPrompt(submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected queued submit command");
        };
        runtime.apply_runner_event(RunnerEvent::QueuedPromptAccepted { prompt: submission });

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new("missing API key")));
        runtime.apply_runner_event(RunnerEvent::Done);

        assert!(runtime.queued_prompts.is_empty());
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());
        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert!(!runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));

        runtime.state_mut().set_input("after failure");
        assert_eq!(
            runtime.handle_input_action(InputAction::Submit).unwrap(),
            Some(RuntimeCommand::SubmitPrompt("after failure".into()))
        );
    }

    #[test]
    fn old_error_done_before_queued_prompt_accept_does_not_consume_handoff() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up 1");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("first queue succeeds");
        runtime.state_mut().set_input("follow up 2");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("second queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);
        let Some(RuntimeCommand::SubmitPrompt(first_submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected first queued submit command");
        };

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new("old turn failed")));
        runtime.apply_runner_event(RunnerEvent::ToolBatchFinished);

        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 1".to_string(), "follow up 2".to_string()]
        );
        assert_eq!(
            runtime.queued_prompt_lifecycle.dispatched_prompt(),
            Some("follow up 1")
        );
        assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(!runtime.queued_prompt_lifecycle.is_accepted());
        assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());

        runtime.apply_runner_event(RunnerEvent::QueuedPromptAccepted {
            prompt: first_submission.clone(),
        });
        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::from_submission(
            first_submission,
        )));

        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 2".to_string()]
        );
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(!runtime.queued_prompt_lifecycle.is_accepted());
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && !message.queued)
        ));
    }

    #[test]
    fn old_done_before_queued_prompt_ack_does_not_dispatch_next_prompt() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up 1");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("first queue succeeds");
        runtime.state_mut().set_input("follow up 2");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("second queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);

        let Some(RuntimeCommand::SubmitPrompt(first_submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected first queued submit command");
        };
        runtime.apply_runner_event(RunnerEvent::Done);

        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 1".to_string(), "follow up 2".to_string()]
        );
        assert_eq!(
            runtime.queued_prompt_lifecycle.dispatched_prompt(),
            Some("follow up 1")
        );
        assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && message.queued)
        ));

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::from_submission(
            first_submission,
        )));

        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 2".to_string()]
        );
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1" && !message.queued)
        ));
    }

    #[test]
    fn manual_submit_during_queued_handoff_is_queued_behind_pending_prompt() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up 1");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);
        assert_eq!(
            runtime.take_next_queued_prompt_command(),
            Some(RuntimeCommand::SubmitPrompt("follow up 1".into()))
        );
        runtime.apply_runner_event(RunnerEvent::Done);

        runtime.state_mut().set_input("manual follow up");
        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("manual submit queues");

        assert_eq!(command, None);
        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 1".to_string(), "manual follow up".to_string()]
        );
        assert_eq!(
            runtime.queued_prompt_lifecycle.dispatched_prompt(),
            Some("follow up 1")
        );
        assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "manual follow up" && message.queued)
        ));
    }

    #[test]
    fn queued_prompt_dispatches_after_turn_done() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        runtime.apply_runner_event(RunnerEvent::Done);

        let command = runtime.take_next_queued_prompt_command();

        let Some(RuntimeCommand::SubmitPrompt(submission)) = command else {
            panic!("expected queued submit command");
        };
        assert_eq!(runtime.queued_prompts.len(), 1);
        assert_eq!(runtime.state().phase, AppPhase::Running);
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::User(message)) if message.text == "follow up" && message.queued
        ));

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::from_submission(
            submission,
        )));

        assert_eq!(runtime.queued_prompts.len(), 0);
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::User(message)) if message.text == "follow up" && !message.queued
        ));
        assert_eq!(
            runtime
                .state()
                .timeline
                .items()
                .iter()
                .filter(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up"))
                .count(),
            1
        );
    }

    #[test]
    fn queued_prompts_become_history_on_interruption() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert!(runtime.queued_prompts.is_empty());
        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
    }

    #[test]
    fn interrupted_runner_event_clears_inflight_queued_prompt_handoff_state() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);
        let Some(RuntimeCommand::SubmitPrompt(submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected queued submit command");
        };
        runtime.apply_runner_event(RunnerEvent::QueuedPromptAccepted { prompt: submission });

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert!(runtime.queued_prompts.is_empty());
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(!runtime.queued_prompt_lifecycle.is_accepted());
        assert!(!runtime.queued_prompt_lifecycle.failed_after_accept());
        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
    }

    #[test]
    fn queued_prompt_accept_does_not_consume_history_until_user_message_arrives() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Done);
        let Some(RuntimeCommand::SubmitPrompt(submission)) =
            runtime.take_next_queued_prompt_command()
        else {
            panic!("expected queued submit command");
        };

        runtime.apply_runner_event(RunnerEvent::QueuedPromptAccepted {
            prompt: submission.clone(),
        });

        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up".to_string()]
        );
        assert_eq!(
            runtime.queued_prompt_lifecycle.dispatched_prompt(),
            Some("follow up")
        );
        assert!(runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(runtime.queued_prompt_lifecycle.is_accepted());
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)
        ));

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::from_submission(
            submission,
        )));

        assert!(runtime.queued_prompts.is_empty());
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
        assert!(!runtime.queued_prompt_lifecycle.is_accepted());
        assert!(runtime.state().timeline.items().iter().any(
            |item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && !message.queued)
        ));
    }

    #[test]
    fn queued_prompt_does_not_dispatch_after_single_tool_finished() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::ToolFinished(ToolFinishedEvent::new(
            "tool-1",
            "fs__read",
            "read completed",
            ToolOutcome::Success,
        )));

        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert_eq!(runtime.queued_prompt_lifecycle.dispatched_prompt(), None);
        assert!(!runtime.queued_prompt_lifecycle.has_inflight_handoff());
    }

    #[test]
    fn queued_prompt_dispatches_after_tool_batch_finished() {
        let mut runtime = runtime();
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::ToolFinished(ToolFinishedEvent::new(
            "tool-1",
            "fs__read",
            "read completed",
            ToolOutcome::Success,
        )));
        assert_eq!(runtime.take_next_queued_prompt_command(), None);

        runtime.apply_runner_event(RunnerEvent::ToolBatchFinished);

        assert_eq!(
            runtime.take_next_queued_prompt_command(),
            Some(RuntimeCommand::SubmitPrompt("follow up".into()))
        );
        assert_eq!(
            runtime.queued_prompt_lifecycle.dispatched_prompt(),
            Some("follow up")
        );
    }

    #[test]
    fn non_terminal_error_does_not_drop_or_dispatch_queued_prompt() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("follow up");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new(
            "failed to view child transcript",
        )));

        assert_eq!(runtime.queued_prompts.len(), 1);
        assert_eq!(runtime.take_next_queued_prompt_command(), None);
        assert!(runtime
            .state()
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up" && message.queued)));
    }

    #[test]
    fn prompt_after_non_terminal_error_still_queues_until_done() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.runner_turn_active = true;
        runtime.state_mut().set_input("follow up 1");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("first queue succeeds");

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new(
            "failed to view child transcript",
        )));
        runtime.state_mut().set_input("follow up 2");
        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("second queue succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 1".to_string(), "follow up 2".to_string()]
        );

        runtime.apply_runner_event(RunnerEvent::Done);

        assert_eq!(
            runtime.take_next_queued_prompt_command(),
            Some(RuntimeCommand::SubmitPrompt("follow up 1".into()))
        );
        assert_eq!(
            runtime.queued_prompts.iter().cloned().collect::<Vec<_>>(),
            vec!["follow up 1".to_string(), "follow up 2".to_string()]
        );
        assert!(matches!(
            runtime.state().timeline.items().iter().find(|item| matches!(item, TimelineItem::User(message) if message.text == "follow up 1")),
            Some(TimelineItem::User(message)) if message.queued
        ));
    }

    #[test]
    fn notice_runner_event_updates_toast_without_timeline_noise() {
        let mut runtime = runtime();
        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::info("Explorer started")));

        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Explorer started")
        );
        assert!(runtime.state().timeline.items().is_empty());
    }

    #[test]
    fn fast_mode_events_update_the_tui_state() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::FastModeChanged { enabled: true });
        assert!(runtime.state().fast_mode_enabled);

        runtime.apply_runner_event(RunnerEvent::FastModeChanged { enabled: false });
        assert!(!runtime.state().fast_mode_enabled);
    }

    #[test]
    fn model_state_changes_only_after_model_changed_event() {
        let mut runtime = runtime();
        let original_model = runtime.state().model_id.clone();

        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::info(
            "model request accepted",
        )));
        assert_eq!(runtime.state().model_id, original_model);

        runtime.apply_runner_event(RunnerEvent::ModelChanged {
            model_id: "gpt-5.5-mini".into(),
        });
        assert_eq!(runtime.state().model_id, "gpt-5.5-mini");
    }

    #[test]
    fn undo_redo_notices_use_info_toast_without_timeline_noise() {
        for message in [
            "already at the start of session history",
            "no history entry available to redo",
        ] {
            let mut runtime = runtime();
            runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::info(message)));

            let toast = runtime.state().toast().expect("info toast");
            assert_eq!(toast.message, message);
            assert_eq!(toast.kind, ToastKind::Info);
            assert!(runtime.state().timeline.items().is_empty());
        }
    }

    #[test]
    fn child_navigation_prefix_survives_tick_and_routes_arrow_actions() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .replace_session_timeline_from_records(&[TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "parent prompt".into(),
                },
            }]);
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime
            .handle_input_action(InputAction::ChildPrefix)
            .expect("prefix succeeds");
        assert!(runtime.state().child_navigation_prefix);

        runtime
            .handle_input_action(InputAction::Tick)
            .expect("tick succeeds");
        assert!(runtime.state().child_navigation_prefix);

        let command = runtime
            .handle_input_action(InputAction::ChildParent)
            .expect("child parent succeeds");
        assert_eq!(command, None);
        assert!(!runtime.state().child_navigation_prefix);
        assert_eq!(
            runtime.state().transcript_view,
            crate::tui::state::TranscriptViewState::Parent
        );
    }

    #[test]
    fn ctrl_x_down_from_parent_maps_to_first_child_navigation() {
        let mut runtime = runtime();
        runtime
            .handle_input_action(InputAction::ChildPrefix)
            .expect("prefix succeeds");

        let command = runtime
            .handle_input_action(InputAction::ChildFirst)
            .expect("child first succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::First,
                anchor_child_session_id: None
            })
        );
    }

    #[test]
    fn child_view_arrow_navigation_works_without_prefix() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            2,
            1,
        );

        let next = runtime
            .handle_input_action(InputAction::ChildNext)
            .expect("next succeeds");
        assert_eq!(
            next,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::Next,
                anchor_child_session_id: None
            })
        );

        let prev = runtime
            .handle_input_action(InputAction::ChildPrev)
            .expect("prev succeeds");
        assert_eq!(
            prev,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::Prev,
                anchor_child_session_id: None
            })
        );

        let parent = runtime
            .handle_input_action(InputAction::ChildParent)
            .expect("parent succeeds");
        assert_eq!(parent, None);
        assert_eq!(
            runtime.state().transcript_view,
            crate::tui::state::TranscriptViewState::Parent
        );
    }

    #[test]
    fn child_view_ignores_direct_text_edit_actions_until_command_entry_starts() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "fixer",
            0,
            1,
            1,
        );

        runtime
            .handle_input_action(InputAction::Insert('x'))
            .expect("insert succeeds");
        runtime
            .handle_input_action(InputAction::Backspace)
            .expect("backspace succeeds");

        assert!(runtime.state().input_buffer.is_empty());

        runtime
            .handle_input_action(InputAction::Insert('/'))
            .expect("slash succeeds");
        runtime
            .handle_input_action(InputAction::Insert('t'))
            .expect("text insert succeeds once command entry starts");

        assert_eq!(runtime.state().input_buffer, "/t");
    }

    #[test]
    fn child_view_scroll_actions_keep_read_only_child_view_active() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "child response".into(),
                },
            }],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime
            .handle_input_action(InputAction::ScrollDown)
            .expect("scroll down succeeds");
        runtime
            .handle_input_action(InputAction::ScrollUp)
            .expect("scroll up succeeds");

        assert!(runtime.state().transcript_view.is_child());
    }

    #[test]
    fn child_view_mouse_click_keeps_read_only_child_view_active() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "fixer",
            0,
            1,
            1,
        );

        let command = runtime
            .handle_input_action(InputAction::MouseClick)
            .expect("mouse click succeeds");

        assert_eq!(command, None);
        assert!(runtime.state().is_read_only_child_view());
    }

    #[test]
    fn child_view_mouse_wheel_scroll_does_not_exit_child_view() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime
            .handle_input_action(InputAction::MouseScrollUp)
            .expect("mouse scroll up succeeds");
        runtime
            .handle_input_action(InputAction::MouseScrollDown)
            .expect("mouse scroll down succeeds");

        assert!(runtime.state().is_read_only_child_view());
    }

    #[test]
    fn runner_events_continue_updating_parent_timeline_while_viewing_child() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .replace_session_timeline_from_records(&[TranscriptRecord {
                session_id: "parent-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "parent prompt".into(),
                },
            }]);
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "child response".into(),
                },
            }],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        runtime.apply_runner_event(RunnerEvent::ToolStarted(
            crate::tui::events::ToolStartedEvent {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                summary: "run ls".into(),
                arguments: Some("ls".into()),
            },
        ));
        runtime.apply_runner_event(RunnerEvent::Done);

        assert!(matches!(
            runtime.state().active_timeline().items().as_ref(),
            [crate::tui::TimelineItem::Assistant(message)] if message.text == "child response"
        ));
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(crate::tui::TimelineItem::Tool(tool)) if tool.call_id == "call-1"
        ));

        runtime.state_mut().restore_parent_timeline_view();

        assert!(matches!(
            runtime.state().active_timeline().items().first(),
            Some(crate::tui::TimelineItem::User(message)) if message.text == "parent prompt"
        ));
        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Tool(tool)) if tool.call_id == "call-1"
        ));
    }

    #[test]
    fn child_navigation_prefix_times_out_after_ticks() {
        let mut runtime = runtime();

        runtime
            .handle_input_action(InputAction::ChildPrefix)
            .expect("prefix succeeds");

        for _ in 0..CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS {
            runtime
                .handle_input_action(InputAction::Tick)
                .expect("tick succeeds");
        }

        assert!(!runtime.state().child_navigation_prefix);
    }

    #[test]
    fn slash_help_is_local_footer_not_agent_prompt() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/help");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert!(runtime.state().input_buffer.is_empty());
        assert!(runtime.submitted_prompts().is_empty());
        assert_eq!(runtime.state().timeline.items().len(), 0);
    }

    #[test]
    fn tree_dispatches_session_history_and_retired_branch_commands_are_invalid() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/branches");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("branches command"),
            None
        );

        runtime.state_mut().set_input("/tree");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("tree command"),
            Some(RuntimeCommand::ShowHistoryTree)
        );
    }

    #[test]
    fn recorder_current_branch_updates_on_resume_checkout_and_new_session() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-branch-sync-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&sessions_dir).expect("create sessions dir");
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("session started");
        recorder.record_user_message("root").expect("root message");
        recorder
            .record_context_branch_created(
                "feature",
                crate::transcript::ROOT_CONTEXT_BRANCH_ID,
                2,
                None,
            )
            .expect("branch created");
        recorder.set_current_context_branch_id(Some("feature".into()));
        recorder
            .record_assistant_message("branch reply")
            .expect("branch message");
        recorder
            .record_context_checkout("feature", 4)
            .expect("checkout metadata");
        let session_id = recorder.session_id().to_string();
        let path = recorder.path().to_path_buf();

        let records = read_records(&path).expect("read records");
        let snapshot =
            transcript_projection::project_session_restore_snapshot(session_id.clone(), records)
                .expect("snapshot");
        let mut reopened =
            TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("reopen recorder");
        sync_recorder_branch(&mut reopened, &snapshot.branch_id);
        assert_eq!(reopened.current_context_branch_id(), Some("feature"));

        sync_recorder_branch(&mut reopened, crate::transcript::ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(reopened.current_context_branch_id(), None);

        let mut fresh = TranscriptRecorder::create(&sessions_dir).expect("fresh recorder");
        fresh.set_current_context_branch_id(Some("temp".into()));
        fresh.set_current_context_branch_id(None);
        assert_eq!(fresh.current_context_branch_id(), None);
    }

    #[test]
    fn slash_compact_routes_to_runtime_command() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/compact");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("compact command succeeds");

        assert_eq!(command, Some(RuntimeCommand::Compact));

        assert_eq!(runtime.state().phase, AppPhase::Running);
    }

    #[test]
    fn compact_noop_notice_remains_visible_after_done() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::info(
            "Nothing to compact yet",
        )));
        runtime.apply_runner_event(RunnerEvent::Done);

        assert!(runtime.state().timeline.items().is_empty());
        assert_eq!(
            runtime.state().toast().map(|toast| toast.message.as_str()),
            Some("Nothing to compact yet")
        );
        assert_eq!(runtime.state().phase, AppPhase::Completed);
    }

    #[test]
    fn tool_output_command_toggles_and_parses_explicit_modes() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/tool-output");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("toggle succeeds");

        assert_eq!(command, None);
        assert!(runtime.state().tool_output_expanded);

        runtime.state_mut().set_input("/tool-output off");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("off succeeds");
        assert!(!runtime.state().tool_output_expanded);

        runtime.state_mut().set_input("/tool-output full");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("full succeeds");
        assert!(runtime.state().tool_output_expanded);
    }

    #[test]
    fn tool_output_command_works_while_running() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("/tool-output on");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("local command succeeds while running");

        assert_eq!(command, None);
        assert!(runtime.state().tool_output_expanded);
    }

    #[test]
    fn tool_output_command_works_in_child_view() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        runtime.state_mut().set_input("/tool-output expanded");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("local command succeeds in child view");

        assert_eq!(command, None);
        assert!(runtime.state().tool_output_expanded);
    }

    #[test]
    fn scrollbar_command_toggles_and_parses_explicit_modes() {
        let mut runtime = runtime();
        assert!(runtime.state().transcript_scrollbar_visible);

        runtime.state_mut().set_input("/scrollbar off");
        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("off succeeds");

        assert_eq!(command, None);
        assert!(!runtime.state().transcript_scrollbar_visible);

        runtime.state_mut().set_input("/scrollbar");
        runtime
            .handle_input_action(InputAction::Submit)
            .expect("toggle succeeds");
        assert!(runtime.state().transcript_scrollbar_visible);
    }

    #[test]
    fn scrollbar_command_persists_preference() {
        let mut runtime = runtime();
        let preferences_dir = runtime.preferences_dir.clone();
        runtime.state_mut().set_input("/scrollbar off");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("scrollbar command succeeds");

        let preferences = TuiPreferences::load_from_dir(&preferences_dir);
        assert!(!preferences.transcript_scrollbar_visible);
    }

    #[test]
    fn scrollbar_command_works_while_running() {
        let mut runtime = runtime();
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().set_input("/scrollbar hide");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("local command succeeds while running");

        assert_eq!(command, None);
        assert!(!runtime.state().transcript_scrollbar_visible);
    }

    #[test]
    fn child_slash_commands_route_to_runtime_navigation_commands() {
        for (input, expected) in [
            ("/child", SharedChildNavigation::First),
            ("/children next", SharedChildNavigation::Next),
            ("/child prev", SharedChildNavigation::Prev),
            ("/parent", SharedChildNavigation::First),
        ] {
            let mut runtime = runtime();
            runtime.state_mut().set_input(input);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("command succeeds");

            let expected = if input == "/parent" {
                Some(RuntimeCommand::ViewParent)
            } else {
                Some(RuntimeCommand::ViewChild {
                    navigation: expected,
                    anchor_child_session_id: None,
                })
            };
            assert_eq!(command, expected, "{input}");
        }
    }

    #[test]
    fn tick_refreshes_active_child_view_from_disk() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-refresh-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("create child");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        child
            .record_session_started("gpt-child")
            .expect("record child start");
        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "running",
                "inspecting",
            )
            .expect("record child result");

        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            sessions_dir.clone(),
            std::env::temp_dir(),
        );
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: child_session_id.clone(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-child".into(),
                },
            }],
            parent_session_id,
            child_session_id.clone(),
            "explorer",
            0,
            1,
            1,
        );

        child
            .record_assistant_message("latest child output")
            .expect("record child message");

        runtime
            .handle_input_action(InputAction::Tick)
            .expect("tick succeeds");

        let metadata = runtime
            .state()
            .child_view_metadata()
            .expect("child metadata");
        assert_eq!(metadata.record_count, 2);
        assert_eq!(metadata.total, 1);
        assert_eq!(runtime.state().phase, AppPhase::Running);
    }

    #[test]
    fn tick_does_not_clobber_live_child_stream_with_disk_records() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-live-refresh-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("create child");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        child
            .record_session_started("gpt-child")
            .expect("record child start");
        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "running",
                "inspecting",
            )
            .expect("record child result");

        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            sessions_dir.clone(),
            std::env::temp_dir(),
        );
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: child_session_id.clone(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-child".into(),
                },
            }],
            parent_session_id,
            child_session_id.clone(),
            "explorer",
            0,
            1,
            1,
        );
        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: child_session_id.clone(),
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::AssistantDelta(crate::tui::events::AssistantDeltaEvent::new(
                "partial stream",
            )),
        });

        child
            .record_tool_call_started("call-1", "shell__exec", serde_json::json!({}))
            .expect("record child tool start");

        runtime
            .handle_input_action(InputAction::Tick)
            .expect("tick succeeds");

        let metadata = runtime
            .state()
            .child_view_metadata()
            .expect("child metadata");
        assert_eq!(metadata.record_count, 1);
        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Assistant(message)) if message.text == "partial stream"
        ));
    }

    #[test]
    fn tick_refreshes_running_child_view_before_parent_result_exists() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-running-child-refresh-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("create child");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        child
            .record_session_started("gpt-child")
            .expect("record child start");

        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            sessions_dir.clone(),
            std::env::temp_dir(),
        );
        runtime.state_mut().phase = AppPhase::Running;
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: child_session_id.clone(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-child".into(),
                },
            }],
            parent_session_id,
            child_session_id.clone(),
            "explorer",
            0,
            1,
            1,
        );

        child
            .record_assistant_message("running child output")
            .expect("record child message");

        runtime
            .handle_input_action(InputAction::Tick)
            .expect("tick succeeds");

        let metadata = runtime
            .state()
            .child_view_metadata()
            .expect("child metadata");
        assert_eq!(metadata.record_count, 2);
        assert_eq!(metadata.index, 0);
        assert_eq!(metadata.total, 1);
        assert_eq!(runtime.state().phase, AppPhase::Running);
    }

    #[test]
    fn child_navigation_dedupes_active_and_completed_child_entries() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-dedupe-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let child = TranscriptRecorder::create(&child_dir).expect("create child");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "completed",
                "done",
            )
            .expect("record child result");

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::First,
            None,
        );

        assert_eq!(selected.as_deref(), Some(child_session_id.as_str()));
        match rx.try_recv().expect("view event") {
            RunnerEvent::ChildSessionViewed { total, .. } => assert_eq!(total, 1),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn child_navigation_uses_durable_child_pool_order() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-active-order-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let completed_child =
            TranscriptRecorder::create(&child_dir).expect("create completed child");
        let active_child = TranscriptRecorder::create(&child_dir).expect("create active child");
        let parent_session_id = parent.session_id().to_string();
        let completed_child_session_id = completed_child.session_id().to_string();
        let active_child_session_id = active_child.session_id().to_string();

        parent
            .record_subagent_started(
                "run-active",
                &parent_session_id,
                "turn-2",
                &active_child_session_id,
                "fixer",
                "still running",
                0,
            )
            .expect("record active child");
        parent
            .record_subagent_result(
                "run-completed",
                &parent_session_id,
                "turn-1",
                &completed_child_session_id,
                "explorer",
                "completed",
                "done",
            )
            .expect("record child result");

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::First,
            None,
        );
        let first_id = selected.expect("first child selected");
        assert!(
            first_id == completed_child_session_id || first_id == active_child_session_id,
            "first child must come from the durable pool"
        );
        match rx.try_recv().expect("view event") {
            RunnerEvent::ChildSessionViewed {
                child_session_id,
                index,
                total,
                pool_ordinal,
                ..
            } => {
                assert_eq!(child_session_id, first_id);
                assert_eq!(index, 0);
                assert_eq!(total, 2);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let selected_next = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Next,
            Some(first_id.as_str()),
        );
        let second_id = selected_next.expect("second child selected");
        assert_ne!(second_id, first_id);
        assert!(
            second_id == completed_child_session_id || second_id == active_child_session_id,
            "next child must come from the durable pool"
        );

        let selected_wrap = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Next,
            Some(second_id.as_str()),
        );
        assert_eq!(selected_wrap.as_deref(), Some(first_id.as_str()));
    }

    #[test]
    fn local_parent_return_then_child_enter_uses_no_stale_anchor() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-anchor-reset-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let first = TranscriptRecorder::create(&child_dir).expect("create first child");
        let second = TranscriptRecorder::create(&child_dir).expect("create second child");
        let parent_session_id = parent.session_id().to_string();
        let first_id = first.session_id().to_string();
        let second_id = second.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &first_id,
                "explorer",
                "completed",
                "first",
            )
            .expect("record first result");
        parent
            .record_subagent_result(
                "run-2",
                &parent_session_id,
                "turn-2",
                &second_id,
                "explorer",
                "completed",
                "second",
            )
            .expect("record second result");

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::First,
            Some(second_id.as_str()),
        );
        assert_eq!(selected.as_deref(), Some(first_id.as_str()));
        match rx.try_recv().expect("view event") {
            RunnerEvent::ChildSessionViewed {
                child_session_id,
                index,
                ..
            } => {
                assert_eq!(child_session_id, first_id);
                assert_eq!(index, 0);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn slash_parent_local_return_then_slash_child_starts_from_first() {
        let mut runtime = runtime();
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session-2",
            "explorer",
            1,
            2,
            1,
        );

        runtime.state_mut().set_input("/parent");
        let parent = runtime
            .handle_input_action(InputAction::Submit)
            .expect("parent command succeeds");
        assert_eq!(parent, None);
        assert_eq!(
            runtime.state().transcript_view,
            crate::tui::state::TranscriptViewState::Parent
        );

        runtime.state_mut().set_input("/child");
        let child = runtime
            .handle_input_action(InputAction::Submit)
            .expect("child command succeeds");
        assert_eq!(
            child,
            Some(RuntimeCommand::ViewChild {
                navigation: SharedChildNavigation::First,
                anchor_child_session_id: None
            })
        );
    }

    #[test]
    fn child_view_left_right_cycle_from_current_visible_child() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-cycle-anchor-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let first = TranscriptRecorder::create(&child_dir).expect("create first child");
        let second = TranscriptRecorder::create(&child_dir).expect("create second child");
        let parent_session_id = parent.session_id().to_string();
        let first_id = first.session_id().to_string();
        let second_id = second.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &first_id,
                "explorer",
                "completed",
                "first",
            )
            .expect("record first");
        parent
            .record_subagent_result(
                "run-2",
                &parent_session_id,
                "turn-2",
                &second_id,
                "explorer",
                "completed",
                "second",
            )
            .expect("record second");

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let next = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Next,
            Some(first_id.as_str()),
        );
        assert_eq!(next.as_deref(), Some(second_id.as_str()));
        let _ = rx.try_recv();

        let prev = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Prev,
            Some(second_id.as_str()),
        );
        assert_eq!(prev.as_deref(), Some(first_id.as_str()));
    }

    #[test]
    fn next_without_anchor_selects_first_and_prev_without_anchor_selects_last() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-child-anchor-none-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let mut parent = TranscriptRecorder::create(&sessions_dir).expect("create parent");
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let first = TranscriptRecorder::create(&child_dir).expect("create first child");
        let second = TranscriptRecorder::create(&child_dir).expect("create second child");
        let parent_session_id = parent.session_id().to_string();
        let first_id = first.session_id().to_string();
        let second_id = second.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &first_id,
                "explorer",
                "completed",
                "first",
            )
            .expect("record first");
        parent
            .record_subagent_result(
                "run-2",
                &parent_session_id,
                "turn-2",
                &second_id,
                "explorer",
                "completed",
                "second",
            )
            .expect("record second");

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let next = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Next,
            None,
        );
        assert_eq!(next.as_deref(), Some(first_id.as_str()));
        let _ = rx.try_recv();

        let prev = crate::session::SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            Some(sessions_dir.as_path()),
            SharedChildNavigation::Prev,
            None,
        );
        assert_eq!(prev.as_deref(), Some(second_id.as_str()));
    }

    #[test]
    fn invalid_child_slash_argument_stays_local_and_shows_usage_hint() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/child sideways");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert!(runtime.submitted_prompts().is_empty());
    }

    #[test]
    fn bare_delegate_without_task_shows_usage() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@fixer");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
    }

    #[test]
    fn delegate_explorer_routes_to_runtime_command() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .show_toast("stale notice", ToastKind::Info);
        runtime
            .state_mut()
            .set_input("@explorer inspect src/agent.rs");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "inspect src/agent.rs".into()
            })
        );

        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::Delegation(item))
                if item.agent_name == "explorer" && item.task == "inspect src/agent.rs"
        ));
        assert!(runtime.state().toast().is_none());
    }

    #[test]
    fn unknown_delegate_shows_error() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@unknown foo");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
    }

    #[test]
    fn delegate_fixer_routes_to_runtime_command() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_input("@fixer wire agent__fixer tool");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::DelegateSubagent {
                agent_name: "fixer".into(),
                task: "wire agent__fixer tool".into()
            })
        );
    }

    #[test]
    fn delegate_oracle_adds_dedicated_delegation_item() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@oracle review src/main.rs");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::DelegateSubagent {
                agent_name: "oracle".into(),
                task: "review src/main.rs".into()
            })
        );
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::Delegation(item))
                if item.agent_name == "oracle" && item.task == "review src/main.rs"
        ));
    }

    #[test]
    fn slash_reasoning_updates_runtime_effort() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/reasoning high");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetReasoningEffort(
                ModelReasoningEffort::High
            ))
        );
        assert_eq!(
            runtime.state().reasoning_effort_label.as_deref(),
            Some("high")
        );
    }

    #[test]
    fn slash_reasoning_without_args_opens_dialog() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_reasoning_effort_label(Some("high".into()));
        runtime.state_mut().set_input("/reasoning");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        let dialog = runtime.state().dialog().expect("dialog should be open");
        assert_eq!(dialog.kind, DialogKind::ReasoningPicker);
        assert_eq!(dialog.title, "Reasoning effort");
        assert_eq!(dialog.selected, 4);
        assert_eq!(dialog.items.len(), 6);
        assert_eq!(dialog.items[0].id, "none");
        assert_eq!(dialog.items[0].label, "Off");
        assert_eq!(dialog.items[5].id, "xhigh");
    }

    #[test]
    fn dialog_accept_switches_selected_reasoning_effort() {
        let mut runtime = runtime();
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            "Reasoning effort",
            None,
            reasoning_dialog_items(&[
                ModelReasoningEffort::None,
                ModelReasoningEffort::Minimal,
                ModelReasoningEffort::Low,
                ModelReasoningEffort::Medium,
                ModelReasoningEffort::High,
                ModelReasoningEffort::Xhigh,
            ]),
        );
        dialog.selected = 0;
        runtime.state_mut().open_dialog(dialog);

        let command = runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("dialog accept succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetReasoningEffort(
                ModelReasoningEffort::None
            ))
        );
        assert!(runtime.state().dialog().is_none());
        assert_eq!(
            runtime.state().reasoning_effort_label.as_deref(),
            Some("off")
        );
    }

    #[test]
    fn configured_reasoning_efforts_limit_dialog_and_cycle_choices() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.6-terra", "GPT-5.6 Terra", "default"),
            rx,
            vec![AvailableModel::with_context_window_and_reasoning(
                "gpt-5.6-terra",
                "GPT-5.6 Terra",
                Some(500_000),
                Some(ModelReasoningEffort::Medium),
                vec![
                    ModelReasoningEffort::None,
                    ModelReasoningEffort::Low,
                    ModelReasoningEffort::Medium,
                    ModelReasoningEffort::Max,
                ],
            )],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime
            .state_mut()
            .set_reasoning_effort_label(Some("medium".into()));
        runtime.state_mut().set_input("/reasoning");

        runtime
            .handle_input_action(InputAction::Submit)
            .expect("dialog opens");
        let dialog = runtime.state().dialog().expect("reasoning dialog");
        assert_eq!(dialog.items.len(), 4);
        assert_eq!(dialog.items[3].id, "max");
        assert_eq!(dialog.selected, 2);

        runtime.state_mut().close_dialog();
        let command = runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("cycle succeeds");
        assert_eq!(
            command,
            Some(RuntimeCommand::SetReasoningEffort(
                ModelReasoningEffort::Max
            ))
        );

        runtime.state_mut().set_input("/reasoning high");
        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("unsupported level stays local");
        assert_eq!(command, None);
    }

    #[test]
    fn ctrl_t_cycles_reasoning_effort() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_reasoning_effort_label(Some("medium".into()));

        let command = runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("cycle succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetReasoningEffort(
                ModelReasoningEffort::High
            ))
        );
        assert_eq!(
            runtime.state().reasoning_effort_label.as_deref(),
            Some("high")
        );
    }

    #[test]
    fn reasoning_controls_stay_local_for_models_without_reasoning() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("plain-model", "Plain Model", "default"),
            rx,
            vec![AvailableModel::with_context_window_and_reasoning(
                "plain-model",
                "Plain Model",
                None,
                None,
                Vec::new(),
            )],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        let command = runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("cycle stays local");
        assert_eq!(command, None);

        runtime.state_mut().set_input("/reasoning");
        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("picker stays local");
        assert_eq!(command, None);
        assert!(runtime.state().dialog().is_none());
    }

    #[test]
    fn slash_permission_without_args_opens_dialog() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/permission");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        let dialog = runtime.state().dialog().expect("dialog should be open");
        assert_eq!(dialog.title, "Permission mode");
        assert_eq!(dialog.selected, 1);
        assert_eq!(dialog.items.len(), 3);
        assert_eq!(dialog.items[0].label, "Safe");
        assert_eq!(dialog.items[2].label, "Solo");
    }

    #[test]
    fn dialog_accept_switches_selected_permission_mode() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::PermissionPicker,
            "Permission mode",
            None,
            vec![
                DialogItem::new("safe", "Safe", Some("Ask before all tools".into())),
                DialogItem::new(
                    "default",
                    "Default",
                    Some("Allow read/preview, ask for risky tools".into()),
                ),
                DialogItem::new(
                    "solo",
                    "Solo",
                    Some("Allow write and command tools without asking".into()),
                ),
            ],
        ));
        runtime
            .state_mut()
            .dialog_mut()
            .expect("dialog exists")
            .selected = 2;

        let command = runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("dialog accept succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetPermissionMode(PermissionMode::Solo))
        );
        assert!(runtime.state().dialog().is_none());
        assert_eq!(runtime.state().permission_mode_label, "solo");
    }

    #[test]
    fn slash_model_emits_command_then_applies_backend_confirmed_model() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![
                AvailableModel::new("gpt-5.5", "GPT-5.5"),
                AvailableModel::new("gpt-5.5-mini", "GPT-5.5 Mini"),
            ],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.state_mut().set_input("/model gpt-5.5-mini");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetModel("gpt-5.5-mini".into()))
        );
        assert_eq!(runtime.state().model_id, "gpt-5.5");
        assert_eq!(runtime.state().model_label, "GPT-5.5");

        runtime.apply_runner_event(RunnerEvent::ModelChanged {
            model_id: "gpt-5.5-mini".into(),
        });

        assert_eq!(runtime.state().model_id, "gpt-5.5-mini");
        assert_eq!(runtime.state().model_label, "GPT-5.5 Mini");
    }

    #[test]
    fn slash_model_without_args_opens_dialog() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![
                AvailableModel::new("gpt-5.5", "GPT-5.5"),
                AvailableModel::new("gpt-5.5-mini", "GPT-5.5 Mini"),
            ],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.state_mut().set_input("/model");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        let dialog = runtime.state().dialog().expect("dialog should be open");
        assert_eq!(dialog.title, "Select model");
        assert_eq!(dialog.selected, 0);
        assert_eq!(dialog.items.len(), 2);
        assert_eq!(dialog.items[1].label, "GPT-5.5 Mini");
    }

    #[test]
    fn slash_resume_with_id_returns_runner_command() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/resume abc123");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::ResumeSession("abc123".into()))
        );
        assert!(runtime.state().input_buffer.is_empty());
    }

    #[test]
    fn slash_new_returns_runner_command() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/new");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, Some(RuntimeCommand::NewSession));
        assert!(runtime.state().input_buffer.is_empty());
    }

    #[test]
    fn slash_resume_without_id_opens_session_dialog() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-resume-dialog-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("record session started");
        recorder
            .record_user_message("restore me")
            .expect("record user message");

        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            sessions_dir,
            std::env::temp_dir(),
        );
        runtime.state_mut().set_input("/resume");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        let dialog = runtime.state().dialog().expect("dialog should be open");
        assert_eq!(dialog.kind, DialogKind::SessionPicker);
        assert_eq!(dialog.items.len(), 1);
        assert_eq!(dialog.items[0].label, "restore me");
    }

    #[test]
    fn empty_session_path_only_matches_session_started_only_transcripts() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-empty-session-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("record session started");
        let path = recorder.path().to_path_buf();

        assert_eq!(empty_session_path(&path), Some(path.clone()));

        recorder
            .record_user_message("now non-empty")
            .expect("record user message");
        assert_eq!(empty_session_path(&path), None);
    }

    #[test]
    fn remove_current_empty_session_deletes_session_started_only_transcript() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-remove-current-empty-session-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("record session started");
        let path = recorder.path().to_path_buf();
        let recorder = Arc::new(StdMutex::new(recorder));

        assert!(remove_current_empty_session(&recorder).expect("remove empty session"));
        assert!(!path.exists());
    }

    #[test]
    fn dialog_accept_selected_session_returns_resume_command() {
        let mut runtime = runtime();
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::SessionPicker,
            "Resume session",
            None,
            vec![DialogItem::new("session-1", "old task", None)],
        ));

        let command = runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("dialog accept succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::ResumeSession("session-1".into()))
        );
        assert!(runtime.state().dialog().is_none());
    }

    #[test]
    fn initial_session_metadata_loads_persisted_title() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-terminal-title-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create transcript");
        recorder
            .record_session_title("Startup title")
            .expect("record title");
        let session_id = recorder.session_id().to_string();

        let (loaded_session_id, title) =
            initial_session_metadata(&Arc::new(StdMutex::new(recorder))).expect("load metadata");

        assert_eq!(loaded_session_id, session_id);
        assert_eq!(title.as_deref(), Some("Startup title"));
    }

    #[test]
    fn terminal_title_uses_latest_persisted_session_title() {
        let records = vec![
            TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionTitle {
                    title: "First title".into(),
                },
            },
            TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 2,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionTitle {
                    title: "Latest title".into(),
                },
            },
        ];

        assert_eq!(
            session_title_from_records(&records).as_deref(),
            Some("Latest title")
        );
    }

    #[test]
    fn terminal_title_formats_static_and_active_states() {
        assert_eq!(format_terminal_title(None, None), "LetCode");
        assert_eq!(
            format_terminal_title(Some("Fix startup"), None),
            "LetCode|Fix startup"
        );
        assert_eq!(
            format_terminal_title(Some("Fix startup"), Some(1)),
            "lEtcode|Fix startup"
        );
    }

    #[test]
    fn terminal_title_animation_uses_every_frame_in_order() {
        let titles = (0..TERMINAL_TITLE_ACTIVE_FRAMES.len())
            .map(|frame| format_terminal_title(Some("Work"), Some(frame)))
            .collect::<Vec<_>>();

        assert_eq!(
            titles,
            vec![
                "Letcode|Work",
                "lEtcode|Work",
                "leTcode|Work",
                "letCode|Work",
                "letcOde|Work",
                "letcoDe|Work",
                "letcodE|Work",
            ]
        );
    }

    #[test]
    fn terminal_title_animation_advances_every_six_ticks() {
        let mut runtime = runtime();
        runtime.state_mut().session_id = Some("session-1".into());
        runtime.session_title = Some("Current session".into());
        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("work")));

        for tick in 1..TERMINAL_TITLE_TICKS_PER_FRAME {
            assert_eq!(runtime.terminal_title(), "Letcode|Current session");
            runtime
                .handle_input_action(InputAction::Tick)
                .unwrap_or_else(|error| panic!("tick {tick} succeeds: {error}"));
        }
        assert_eq!(runtime.terminal_title(), "Letcode|Current session");
        runtime
            .handle_input_action(InputAction::Tick)
            .expect("sixth tick succeeds");
        assert_eq!(runtime.terminal_title(), "lEtcode|Current session");
    }

    #[test]
    fn terminal_title_stops_spinning_after_terminal_events() {
        let mut runtime = runtime();
        runtime.state_mut().session_id = Some("session-1".into());
        runtime.session_title = Some("Current session".into());
        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("work")));
        assert_eq!(runtime.terminal_title(), "Letcode|Current session");

        runtime.apply_runner_event(RunnerEvent::Done);
        assert_eq!(runtime.terminal_title(), "LetCode|Current session");

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new(
            "work again",
        )));
        runtime.apply_runner_event(RunnerEvent::Interrupted);
        assert_eq!(runtime.terminal_title(), "LetCode|Current session");
    }

    #[test]
    fn session_title_update_applies_only_to_current_session() {
        let mut runtime = runtime();
        runtime.state_mut().session_id = Some("session-1".into());

        runtime.apply_runner_event(RunnerEvent::SessionTitleUpdated {
            session_id: "session-2".into(),
            title: "Other session".into(),
        });
        assert_eq!(runtime.session_title, None);

        runtime.apply_runner_event(RunnerEvent::SessionTitleUpdated {
            session_id: "session-1".into(),
            title: "Current session".into(),
        });
        assert_eq!(runtime.session_title.as_deref(), Some("Current session"));
    }

    #[test]
    fn session_dialog_item_prefers_persisted_title() {
        let item = session_dialog_item(&SessionSummary {
            session_id: "session-1".into(),
            record_count: 3,
            first_timestamp_ms: Some(0),
            last_timestamp_ms: Some(0),
            model: Some("gpt-test".into()),
            title: Some("Fix startup crash".into()),
            last_user_summary: Some("help debug startup".into()),
            last_assistant_summary: Some("checked logs".into()),
        });

        assert_eq!(item.label, "Fix startup crash");
    }

    #[test]
    fn session_resumed_event_replaces_timeline_not_appends() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new("current session notice"));

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "session-1".into(),
            branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            messages: vec![crate::agent::ConversationMessage {
                role: crate::agent::ConversationRole::User,
                content: "old prompt".into(),
            }],
            records: vec![crate::transcript::TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: crate::transcript::TranscriptEvent::UserMessage {
                    content: "old prompt".into(),
                },
            }],
            evidence_count: 2,
            model_id: None,
            token_usage: None,
            runtime_context: event_context("session-1", 1),
        });

        assert!(matches!(
            runtime.state().timeline.items().first(),
            Some(crate::tui::TimelineItem::User(message)) if message.text == "old prompt"
        ));

        assert!(runtime.state().active_session);
    }

    #[test]
    fn session_resumed_event_restores_persisted_terminal_title() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "session-1".into(),
            branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            messages: Vec::new(),
            records: vec![TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionTitle {
                    title: "Resume this session".into(),
                },
            }],
            evidence_count: 0,
            model_id: None,
            token_usage: None,
            runtime_context: event_context("session-1", 1),
        });

        assert_eq!(
            runtime.session_title.as_deref(),
            Some("Resume this session")
        );
    }

    #[test]
    fn session_resumed_event_restores_recorded_model() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![
                AvailableModel::with_context_window("gpt-5.5", "GPT-5.5", Some(128_000)),
                AvailableModel::with_context_window("gpt-5.5-mini", "GPT-5.5 Mini", Some(64_000)),
            ],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "session-1".into(),
            branch_id: "feature-a".into(),
            messages: Vec::new(),
            records: vec![crate::transcript::TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: crate::transcript::TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature-a".into(),
                    parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 0,
                    label: None,
                },
            }],
            evidence_count: 0,
            model_id: Some("gpt-5.5-mini".into()),
            token_usage: Some(TokenUsageEvent::new(12_345, 64_000)),
            runtime_context: event_context("session-1", 1),
        });

        assert_eq!(runtime.state().model_id, "gpt-5.5-mini");
        assert_eq!(runtime.state().current_context_branch, "feature-a");
        assert_eq!(runtime.state().model_label, "GPT-5.5 Mini");
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.context_window_tokens),
            Some(64_000)
        );
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.used_tokens),
            Some(12_345)
        );
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .and_then(|usage| usage.cache_report.as_ref()),
            None
        );
    }

    #[test]
    fn token_usage_output_counts_current_turn_not_transcript() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("first")));
        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(1_000, 10_000, 1_000, 0, 0)
                .with_cache_report(Some(cache_report(None))),
        ));
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.output_tokens),
            Some(0)
        );

        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(1_200, 10_000, 1_000, 200, 0)
                .with_cache_report(Some(cache_report(Some(20)))),
        ));
        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(1_800, 10_000, 1_500, 300, 50)
                .with_cache_report(Some(cache_report(Some(50)))),
        ));
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.output_tokens),
            Some(500)
        );
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .and_then(|usage| usage.cache_report.as_ref())
                .and_then(|report| report.actual_cached_tokens),
            Some(50)
        );

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("second")));
        runtime.apply_runner_event(RunnerEvent::TokenUsage(TokenUsageEvent::with_breakdown(
            2_000, 10_000, 2_000, 0, 0,
        )));
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.output_tokens),
            Some(0)
        );
    }

    #[test]
    fn session_token_usage_replaces_footer_usage_and_resets_turn_output() {
        let mut runtime = runtime();
        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(9_000, 10_000, 8_000, 700, 300)
                .with_cache_report(Some(cache_report(Some(300)))),
        ));

        runtime.apply_runner_event(RunnerEvent::SessionTokenUsage(
            TokenUsageEvent::with_breakdown(1_200, 10_000, 1_200, 0, 0),
        ));

        let usage = runtime
            .state()
            .model_token_usage
            .as_ref()
            .expect("replacement token usage state");
        assert_eq!(usage.used_tokens, 1_200);
        assert_eq!(usage.input_tokens, 1_200);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cached_tokens, 0);
        assert_eq!(usage.cache_report, None);

        runtime.apply_runner_event(RunnerEvent::TokenUsage(TokenUsageEvent::with_breakdown(
            1_210, 10_000, 1_200, 10, 0,
        )));
        assert_eq!(
            runtime
                .state()
                .model_token_usage
                .as_ref()
                .map(|usage| usage.output_tokens),
            Some(10)
        );
    }

    #[test]
    fn later_token_usage_iteration_replaces_the_entire_cache_report() {
        let mut runtime = runtime();
        let first = cache_report(Some(40));
        let second = CacheUsageReport {
            configured: true,
            hint_serialized: false,
            retention_sent: None,
            stable_prefix_segments: 0,
            stable_prompt_tokens: 0,
            volatile_prompt_tokens: 900,
            cacheable_prefix_tokens: 0,
            stable_after_boundary_tokens: 0,
            local_prefix_fingerprint: None,
            routing_key: None,
            actual_cached_tokens: Some(7),
        };

        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(400, 10_000, 400, 100, 40)
                .with_cache_report(Some(first)),
        ));
        runtime.apply_runner_event(RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(907, 10_000, 900, 200, 7)
                .with_cache_report(Some(second.clone())),
        ));

        let usage = runtime
            .state()
            .model_token_usage
            .as_ref()
            .expect("token usage state");
        assert_eq!(usage.output_tokens, 300);
        assert_eq!(usage.input_tokens, 900);
        assert_eq!(usage.cached_tokens, 7);
        assert_eq!(usage.cache_report.as_ref(), Some(&second));
    }

    #[test]
    fn session_started_event_clears_timeline_for_new_session() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new("current session notice"));

        runtime.apply_runner_event(RunnerEvent::SessionStarted {
            session_id: "new-session".into(),
            records: Vec::new(),
            runtime_context: event_context("new-session", 1),
        });

        assert_eq!(runtime.state().timeline.items().len(), 0);
        assert!(!runtime.state().active_session);
        assert!(runtime.state().show_dashboard());
        assert_eq!(runtime.session_title, None);
    }

    #[test]
    fn session_started_event_restores_persisted_terminal_title() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::SessionStarted {
            session_id: "new-session".into(),
            records: vec![TranscriptRecord {
                session_id: "new-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionTitle {
                    title: "New session title".into(),
                },
            }],
            runtime_context: event_context("new-session", 1),
        });

        assert_eq!(runtime.session_title.as_deref(), Some("New session title"));
    }

    #[test]
    fn invalid_lifecycle_timeline_does_not_clear_parent_permission() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(tx),
        });
        let timeline_len = runtime.state().timeline.items().len();

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "new-session".into(),
            branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            messages: Vec::new(),
            records: vec![TranscriptRecord {
                session_id: "wrong-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "malformed lifecycle timeline".into(),
                },
            }],
            evidence_count: 0,
            model_id: None,
            token_usage: None,
            runtime_context: event_context("new-session", 1),
        });

        assert!(runtime.pending_permission_handle().is_some());
        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
        assert_eq!(runtime.state().timeline.items().len(), timeline_len);
    }

    #[test]
    fn dialog_accept_switches_selected_model() {
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![
                AvailableModel::new("gpt-5.5", "GPT-5.5"),
                AvailableModel::new("gpt-5.5-mini", "GPT-5.5 Mini"),
            ],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.state_mut().open_dialog(DialogState::new(
            DialogKind::ModelPicker,
            "Switch model",
            None,
            vec![
                DialogItem::new("gpt-5.5", "GPT-5.5", Some("gpt-5.5 · current".into())),
                DialogItem::new("gpt-5.5-mini", "GPT-5.5 Mini", Some("gpt-5.5-mini".into())),
            ],
        ));
        runtime
            .state_mut()
            .dialog_mut()
            .expect("dialog exists")
            .selected = 1;

        let command = runtime
            .handle_input_action(InputAction::DialogAccept)
            .expect("dialog accept succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetModel("gpt-5.5-mini".into()))
        );
        assert!(runtime.state().dialog().is_none());
        assert_eq!(runtime.state().model_id, "gpt-5.5-mini");
        assert_eq!(runtime.state().model_label, "GPT-5.5 Mini");
    }

    #[test]
    fn slash_submit_accepts_partial_match_before_execution() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/per");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("submit succeeds");

        assert_eq!(command, None);
        assert_eq!(runtime.state().input_buffer, "/permission ");
        assert!(runtime.state().slash_panel_is_open());
    }

    #[test]
    fn slash_panel_navigation_accept_and_dismiss_work() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/per");

        runtime
            .handle_input_action(InputAction::SlashPanelNext)
            .expect("next succeeds");
        assert_eq!(runtime.state().slash_panel_selected, 0);

        runtime
            .handle_input_action(InputAction::SlashPanelAccept)
            .expect("accept succeeds");
        assert_eq!(runtime.state().input_buffer, "/permission ");

        runtime
            .handle_input_action(InputAction::SlashPanelDismiss)
            .expect("dismiss succeeds");
        assert!(!runtime.state().slash_panel_is_open());
        assert_eq!(runtime.state().input_buffer, "/permission ");
    }

    #[test]
    fn runner_permission_events_update_state_and_handle() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        let handle = RunnerPermissionRequest::new(tx);
        runtime
            .state_mut()
            .show_toast("stale notice", ToastKind::Info);

        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: handle.clone(),
        });

        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
        assert!(runtime.pending_permission_handle().is_some());
        assert!(runtime.state().toast().is_none());

        runtime.apply_runner_event(RunnerEvent::PermissionResolved(
            PermissionResolutionEvent::approved("call-1"),
        ));

        assert!(runtime.pending_permission_handle().is_none());
        assert_eq!(runtime.state().pending_permission, None);
        let permission = runtime
            .state()
            .timeline
            .items()
            .iter()
            .find_map(|item| match item {
                crate::tui::TimelineItem::Permission(permission) => Some(permission),
                _ => None,
            })
            .expect("permission item exists");
        assert_eq!(
            permission.status,
            crate::tui::PermissionPromptStatus::Approved
        );
    }

    #[test]
    fn child_session_viewed_does_not_clear_runtime_pending_permission() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        let handle = RunnerPermissionRequest::new(tx);

        runtime.apply_runner_event(RunnerEvent::ChildPermissionRequested {
            child_session_id: "child-session".into(),
            agent_name: Some("explorer".into()),
            parent_tool_call_id: Some("parent-call".into()),
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle,
        });
        runtime.apply_runner_event(RunnerEvent::ChildSessionViewed {
            parent_session_id: "parent-session".into(),
            child_session_id: "child-session".into(),
            agent_name: "explorer".into(),
            index: 0,
            total: 1,
            pool_ordinal: 1,
            records: vec![],
            runtime_context: event_context("child-session", 1),
        });

        assert!(runtime.pending_permission_handle().is_some());
        assert!(runtime.state().pending_permission.is_some());
        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
    }

    #[tokio::test]
    async fn approve_and_deny_actions_respond_through_pending_handle() {
        let mut approve_runtime = runtime();
        let (approve_tx, approve_rx) = oneshot::channel();
        approve_runtime
            .permission_lifecycle
            .begin_parent(
                PermissionRequestEvent::new("call-a", "shell__exec", "ls"),
                RunnerPermissionRequest::new(approve_tx),
            )
            .expect("seed pending parent permission");
        approve_runtime.reproject_pending_permission();

        approve_runtime
            .handle_input_action(InputAction::ApprovePermission)
            .expect("approve succeeds");
        assert_eq!(
            approve_rx.await.expect("approval received"),
            crate::tui::PermissionResponse::AllowOnce
        );
        assert!(approve_runtime.pending_permission_handle().is_none());

        let mut deny_runtime = runtime();
        let (deny_tx, deny_rx) = oneshot::channel();
        deny_runtime
            .permission_lifecycle
            .begin_parent(
                PermissionRequestEvent::new("call-b", "shell__exec", "rm"),
                RunnerPermissionRequest::new(deny_tx),
            )
            .expect("seed pending parent permission");
        deny_runtime.reproject_pending_permission();

        deny_runtime
            .handle_input_action(InputAction::DenyPermission)
            .expect("deny succeeds");
        assert_eq!(
            deny_rx.await.expect("denial received"),
            crate::tui::PermissionResponse::Deny
        );
        assert!(deny_runtime.pending_permission_handle().is_none());
    }

    #[tokio::test]
    async fn child_permission_request_survives_view_switch_and_can_be_approved() {
        let mut runtime = runtime();
        let (tx, rx) = oneshot::channel();

        runtime.apply_runner_event(RunnerEvent::ChildPermissionRequested {
            child_session_id: "child-session".into(),
            agent_name: Some("explorer".into()),
            parent_tool_call_id: Some("parent-call".into()),
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(tx),
        });
        runtime.state_mut().restore_parent_timeline_view();

        runtime
            .handle_input_action(InputAction::ApprovePermission)
            .expect("approve succeeds");

        assert_eq!(
            rx.await.expect("approval received"),
            PermissionResponse::AllowOnce
        );
        assert!(runtime.pending_permission_handle().is_none());
        assert!(runtime.state().pending_permission.is_some());
    }

    #[test]
    fn non_terminal_runner_error_does_not_clear_pending_permission() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(tx),
        });

        runtime.apply_runner_event(RunnerEvent::Error(ErrorEvent::new(
            "failed to view child transcript",
        )));

        assert!(runtime.pending_permission_handle().is_some());
        assert!(runtime.state().pending_permission.is_some());
        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
    }

    #[tokio::test]
    async fn second_permission_request_is_denied_without_replacing_active_one() {
        let mut runtime = runtime();
        let (first_tx, _first_rx) = oneshot::channel();
        let (second_tx, second_rx) = oneshot::channel();

        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(first_tx),
        });
        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-2", "fs__write", "write file"),
            handle: RunnerPermissionRequest::new(second_tx),
        });

        assert_eq!(
            second_rx.await.expect("denial received"),
            PermissionResponse::Deny
        );
        assert_eq!(
            runtime
                .state()
                .pending_permission
                .as_ref()
                .map(|permission| permission.call_id.as_str()),
            Some("call-1")
        );
        assert_eq!(runtime.permission_lifecycle.child_session_id(), None);

        runtime.apply_runner_event(RunnerEvent::PermissionResolved(
            PermissionResolutionEvent::denied("call-2", None),
        ));
        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "other-child".into(),
            agent_name: None,
            parent_tool_call_id: None,
            event: AppEvent::Interrupted,
        });

        assert!(runtime.pending_permission_handle().is_some());
        assert_eq!(
            runtime
                .state()
                .pending_permission
                .as_ref()
                .map(|permission| permission.call_id.as_str()),
            Some("call-1")
        );
    }

    #[test]
    fn draining_runner_events_applies_shared_update_path() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        tx.send(RunnerEvent::UserMessage(UserMessageEvent::new("hello")))
            .expect("send user event");
        tx.send(RunnerEvent::PermissionResolved(PermissionResolutionEvent {
            call_id: "call-z".into(),
            decision: PermissionDecision::Denied,
            reason: Some("no".into()),
        }))
        .expect("send permission event");

        runtime.try_drain_runner_events();

        assert_eq!(runtime.state().timeline.items().len(), 2);
    }

    #[test]
    fn draining_runner_events_is_bounded_so_input_can_make_progress() {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        for index in 0..300 {
            tx.send(RunnerEvent::UserMessage(UserMessageEvent::new(format!(
                "message-{index}"
            ))))
            .expect("queue runner event");
        }

        runtime.try_drain_runner_events();

        assert!(runtime.runner_rx.try_recv().is_ok());
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Interrupt)
                .expect("input is processed after bounded drain"),
            None
        );
    }

    #[test]
    fn resumed_session_restores_latest_todo_state_from_records() {
        let mut runtime = runtime();
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::AutoContinueChanged {
                    state: AutoContinueState {
                        enabled: true,
                        max_continuations: 2,
                    },
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::TodoSnapshot {
                    items: vec![TodoItem {
                        id: "t1".into(),
                        content: "inspect".into(),
                        status: TodoStatus::InProgress,
                    }],
                },
            },
        ];

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "s".into(),
            branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
            messages: Vec::new(),
            records,
            evidence_count: 0,
            model_id: None,
            token_usage: None,
            runtime_context: event_context("s", 2),
        });

        let todo = runtime
            .state()
            .latest_todo
            .as_ref()
            .expect("todo state restored");
        assert_eq!(todo.items.len(), 1);
        assert_eq!(todo.items[0].status, TodoStatus::InProgress);
        assert!(todo.auto_continue.enabled);
    }

    const RUNNER_INTEGRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

    enum ControlledSseResponse {
        Blocked(String),
        Immediate(String),
    }

    struct ControlledSseServer {
        base_url: String,
        requests: mpsc::UnboundedReceiver<usize>,
        release: Arc<Notify>,
        task: JoinHandle<()>,
    }

    impl ControlledSseServer {
        async fn expect_request(&mut self, expected: usize) {
            let request = timeout(RUNNER_INTEGRATION_TIMEOUT, self.requests.recv())
                .await
                .expect("timed out waiting for SSE request")
                .expect("SSE server stopped before the expected request");
            assert_eq!(request, expected);
        }

        async fn finish(self) {
            self.release.notify_waiters();
            self.release.notify_one();
            timeout(RUNNER_INTEGRATION_TIMEOUT, self.task)
                .await
                .expect("SSE server should finish")
                .expect("SSE server task should not panic");
        }

        async fn abort(self) {
            self.task.abort();
            let _ = self.task.await;
        }
    }

    fn complete_http_request_len(request: &[u8]) -> Option<usize> {
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&request[..header_end])
            .expect("test client sends UTF-8 HTTP headers");
        let content_length = headers
            .lines()
            .find_map(|header| {
                header
                    .split_once(':')
                    .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .map(|(_, value)| {
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("test client sends a numeric content length")
                    })
            })
            .unwrap_or(0);
        Some(header_end + 4 + content_length)
    }

    async fn read_complete_http_request(socket: &mut tokio::net::TcpStream) {
        let mut request = Vec::new();
        loop {
            if complete_http_request_len(&request).is_some_and(|length| request.len() >= length) {
                return;
            }
            let read = socket
                .read_buf(&mut request)
                .await
                .expect("server reads request");
            assert_ne!(read, 0, "test client closed before completing its request");
        }
    }

    async fn spawn_controlled_sse_server(
        responses: Vec<ControlledSseResponse>,
    ) -> ControlledSseServer {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server has local address");
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let server_release = Arc::clone(&release);
        let task = tokio::spawn(async move {
            for (index, response) in responses.into_iter().enumerate() {
                let (mut socket, _) = listener.accept().await.expect("server accepts request");
                read_complete_http_request(&mut socket).await;
                match response {
                    ControlledSseResponse::Blocked(body) => {
                        socket
                            .write_all(
                                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                            )
                            .await
                            .expect("server writes SSE headers");
                        socket.flush().await.expect("server flushes SSE headers");
                        let _ = request_tx.send(index);
                        server_release.notified().await;
                        let _ = socket.write_all(body.as_bytes()).await;
                        let _ = socket.shutdown().await;
                    }
                    ControlledSseResponse::Immediate(body) => {
                        let _ = request_tx.send(index);
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        socket
                            .write_all(response.as_bytes())
                            .await
                            .expect("server writes SSE response");
                        socket.shutdown().await.expect("server closes SSE response");
                    }
                }
            }
        });
        ControlledSseServer {
            base_url: format!("http://{address}"),
            requests: request_rx,
            release,
            task,
        }
    }

    fn compaction_checkpoint(next_step: &str) -> String {
        format!(
            "## Progress\n### Done\n- completed work\n### In Progress\n- continue execution\n### Blocked\n- 无\n## Key Decisions\n- resolved scope\n## Validation\n- pending\n## File Operations\n### Read\n- 无\n### Modified\n- 无\n## Next Steps\n- {next_step}\n## Critical Context\n- durable workflow facts"
        )
    }

    fn responses_sse_body(text: &str) -> String {
        let response = serde_json::json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "r-test", "object": "response", "created_at": 1,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                "model": "m1", "output": [{
                    "type": "message", "id": "m-test", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }],
                "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
                "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
                "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
                "usage": {
                    "input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2
                },
                "user": null, "metadata": {}
            }
        });
        let response = serde_json::to_string(&response).expect("SSE response serializes");
        format!("data: {response}\n\ndata: [DONE]\n\n")
    }

    fn test_transcript(
        name: &str,
        history: Vec<(String, String)>,
    ) -> (PathBuf, Arc<StdMutex<TranscriptRecorder>>) {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-tui-runner-interrupt-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let mut recorder = TranscriptRecorder::create(&sessions_dir).expect("create transcript");
        recorder
            .record_session_started("m1")
            .expect("record session start");
        // Avoid a title-generation side request in tests that exercise a prompt.
        recorder
            .record_session_title("runner interrupt test")
            .expect("record title");
        for (user, assistant) in history {
            recorder
                .record_user_message(user)
                .expect("record user message");
            recorder
                .record_assistant_message(assistant)
                .expect("record assistant message");
        }
        (sessions_dir, Arc::new(StdMutex::new(recorder)))
    }

    fn integration_agent(base_url: String, m1_input_limit_tokens: u64) -> Agent<OpenAIConfig> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        let metadata = |input_limit_tokens: u64| ModelRequestMetadata {
            context_window: Some(input_limit_tokens.saturating_add(1_000)),
            effective_input_limit_tokens: Some(input_limit_tokens),
            max_output_tokens: Some(128),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        };
        agent.set_model_catalog(HashMap::from([
            ("m1".into(), metadata(m1_input_limit_tokens)),
            ("m2".into(), metadata(100_000)),
        ]));
        agent.set_compaction_config(CompactionConfig {
            preserve_recent_tokens: Some(0),
        });
        agent
    }

    fn test_interrupt() -> InterruptRequest {
        InterruptRequest {
            parent_tool_calls: Vec::new(),
            visible_child_session_id: None,
        }
    }

    fn turn_started(turn_id: u64) -> TurnStartedEvent {
        TurnStartedEvent {
            turn_id,
            intent: "test".into(),
            directive: "test turn lifecycle".into(),
            validation_reminder: String::new(),
        }
    }

    struct RunnerHarness {
        control_tx: mpsc::UnboundedSender<RunnerControl>,
        event_rx: mpsc::UnboundedReceiver<RunnerEvent>,
        task: JoinHandle<Agent<OpenAIConfig>>,
    }

    impl RunnerHarness {
        fn send_command(
            &self,
            command: RunnerCommand,
        ) -> std::result::Result<(), mpsc::error::SendError<RunnerControl>> {
            self.control_tx.send(RunnerControl::Command(command))
        }

        fn send_interrupt(
            &self,
            interrupt: InterruptRequest,
        ) -> std::result::Result<(), mpsc::error::SendError<RunnerControl>> {
            self.control_tx.send(RunnerControl::Interrupt(interrupt))
        }
    }

    struct RunnerPollGate {
        ready: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    }

    fn start_runner_harness(
        agent: Agent<OpenAIConfig>,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        sessions_dir: PathBuf,
    ) -> RunnerHarness {
        start_runner_harness_with_poll_gate(agent, transcript, sessions_dir, None)
    }

    fn start_runner_harness_with_poll_gate(
        agent: Agent<OpenAIConfig>,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        sessions_dir: PathBuf,
        poll_gate: Option<RunnerPollGate>,
    ) -> RunnerHarness {
        let (runner_tx, event_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(test_runner_loop(
            agent,
            transcript,
            sessions_dir,
            runner_tx,
            control_rx,
            poll_gate,
        ));
        RunnerHarness {
            control_tx,
            event_rx,
            task,
        }
    }

    async fn start_paused_runner_harness(
        agent: Agent<OpenAIConfig>,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        sessions_dir: PathBuf,
    ) -> (RunnerHarness, oneshot::Sender<()>) {
        let (ready_tx, ready_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let harness = start_runner_harness_with_poll_gate(
            agent,
            transcript,
            sessions_dir,
            Some(RunnerPollGate {
                ready: ready_tx,
                release: release_rx,
            }),
        );
        timeout(RUNNER_INTEGRATION_TIMEOUT, ready_rx)
            .await
            .expect("runner did not reach the control poll gate")
            .expect("runner dropped the control poll gate");
        (harness, release_tx)
    }

    async fn test_runner_loop(
        mut agent: Agent<OpenAIConfig>,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        sessions_dir: PathBuf,
        runner_tx: mpsc::UnboundedSender<RunnerEvent>,
        mut control_rx: mpsc::UnboundedReceiver<RunnerControl>,
        poll_gate: Option<RunnerPollGate>,
    ) -> Agent<OpenAIConfig> {
        let subagent_runtime = SubagentPool::new();
        let runner = AgentRunner::with_transcript(runner_tx.clone(), Arc::clone(&transcript))
            .with_subagent_runtime(subagent_runtime.clone(), sessions_dir.clone());
        let mut deferred_commands = VecDeque::new();

        if let Some(RunnerPollGate { ready, release }) = poll_gate {
            ready.send(()).expect("test releases the control poll gate");
            release
                .await
                .expect("test retains the control poll gate release sender");
        }

        loop {
            let Some(command) =
                next_idle_runner_command(&mut control_rx, &mut deferred_commands).await
            else {
                break;
            };

            match command {
                RunnerCommand::Compact => {
                    run_manual_compaction(
                        &mut agent,
                        &transcript,
                        &runner_tx,
                        &mut control_rx,
                        &mut deferred_commands,
                    )
                    .await;
                }
                RunnerCommand::Prompt(prompt) => {
                    let _ = runner_tx.send(RunnerEvent::QueuedPromptAccepted {
                        prompt: prompt.clone(),
                    });
                    let interrupted = {
                        let run = runner.run_prompt(&mut agent, prompt);
                        tokio::pin!(run);
                        loop {
                            match select_active_runner_operation(
                                &mut control_rx,
                                &mut deferred_commands,
                                run.as_mut(),
                            )
                            .await
                            {
                                ActiveRunnerOperation::Interrupted(interrupt) => {
                                    break Some(interrupt);
                                }
                                ActiveRunnerOperation::Completed(_) => break None,
                                ActiveRunnerOperation::Command(Some(command)) => {
                                    deferred_commands.push_front(command);
                                    break None;
                                }
                                ActiveRunnerOperation::Command(None) => break None,
                            }
                        }
                    };
                    if let Some(interrupt) = interrupted {
                        subagent_runtime.cancel_active();
                        record_interrupt_transcript(&transcript, &interrupt);
                        let _ = rehydrate_agent_from_transcript(&mut agent, &transcript);
                        send_subagent_interrupted(&runner_tx, interrupt.visible_child_session_id);
                    }
                }
                RunnerCommand::DelegateSubagent { agent_name, task } => {
                    let parent_session_id = transcript
                        .lock()
                        .expect("lock transcript")
                        .session_id()
                        .to_string();
                    let input = crate::tool::normalize_subagent_input(
                        &format!("agent__{agent_name}"),
                        &serde_json::json!({ "task": task }),
                    )
                    .expect("delegate input normalizes");
                    let invocation = SubagentInvocation {
                        prompt: input.objective.clone(),
                        input,
                        parent_tool_call_id: None,
                    };
                    let (interrupted, child_started, interrupted_child_session_id) = {
                        let delegate = subagent_runtime.run_named_governed(
                            &agent,
                            &agent_name,
                            invocation,
                            sessions_dir.clone(),
                            parent_session_id,
                            "runner-harness".into(),
                            Some(Arc::clone(&transcript)),
                            Some(crate::session::subagent_event_sender::<OpenAIConfig>(
                                runner_tx.clone(),
                            )),
                        );
                        tokio::pin!(delegate);
                        let mut interrupted = false;
                        let mut child_started = false;
                        let mut interrupted_child_session_id = None;
                        loop {
                            match select_active_runner_operation(
                                &mut control_rx,
                                &mut deferred_commands,
                                delegate.as_mut(),
                            )
                            .await
                            {
                                ActiveRunnerOperation::Interrupted(interrupt) => {
                                    child_started = subagent_runtime.is_running();
                                    interrupted = true;
                                    interrupted_child_session_id = subagent_runtime
                                        .active_child()
                                        .map(|child| child.child_session_id);
                                    if child_started {
                                        subagent_runtime.cancel_active();
                                    }
                                    record_interrupt_transcript(&transcript, &interrupt);
                                    if child_started {
                                        let _ = delegate.await;
                                    }
                                    break;
                                }
                                ActiveRunnerOperation::Completed(result) => {
                                    match result {
                                        Ok(_) => {
                                            let _ = runner_tx.send(RunnerEvent::Done);
                                        }
                                        Err(error) => {
                                            let _ = runner_tx.send(RunnerEvent::Error(
                                                ErrorEvent::new(format!("{error:#}")),
                                            ));
                                            let _ = runner_tx.send(RunnerEvent::Done);
                                        }
                                    }
                                    break;
                                }
                                ActiveRunnerOperation::Command(Some(command)) => {
                                    deferred_commands.push_front(command);
                                    break;
                                }
                                ActiveRunnerOperation::Command(None) => break,
                            }
                        }
                        (interrupted, child_started, interrupted_child_session_id)
                    };
                    if interrupted {
                        if child_started {
                            let _ = rehydrate_agent_from_transcript(&mut agent, &transcript);
                        }
                        send_subagent_interrupted(&runner_tx, interrupted_child_session_id);
                    }
                }
                RunnerCommand::SetModel(model) => {
                    agent.set_model(model);
                }
                #[cfg(test)]
                RunnerCommand::InspectHistory(reply) => {
                    let _ = reply.send(agent.history_for_test().to_vec());
                }
                _ => {}
            }
        }
        agent
    }

    async fn runner_events_until_terminal(harness: &mut RunnerHarness) -> Vec<RunnerEvent> {
        let mut events = Vec::new();
        loop {
            let event = timeout(RUNNER_INTEGRATION_TIMEOUT, harness.event_rx.recv())
                .await
                .expect("timed out waiting for runner event")
                .expect("runner event channel closed before terminal event");
            let terminal = matches!(event, RunnerEvent::Done | RunnerEvent::Interrupted);
            events.push(event);
            if terminal {
                return events;
            }
        }
    }

    async fn runner_events_until_compaction_committed(
        harness: &mut RunnerHarness,
    ) -> Vec<RunnerEvent> {
        let mut events = Vec::new();
        loop {
            let event = timeout(RUNNER_INTEGRATION_TIMEOUT, harness.event_rx.recv())
                .await
                .expect("timed out waiting for compaction event")
                .expect("runner event channel closed before compaction commit");
            let committed = matches!(event, RunnerEvent::CompactionCommitted { .. });
            events.push(event);
            if committed {
                return events;
            }
        }
    }

    async fn inspect_runner_history(harness: &RunnerHarness) -> Vec<HistoryItem> {
        let (reply_tx, reply_rx) = oneshot::channel();
        harness
            .send_command(RunnerCommand::InspectHistory(reply_tx))
            .expect("runner accepts history inspection");
        timeout(RUNNER_INTEGRATION_TIMEOUT, reply_rx)
            .await
            .expect("timed out waiting for history inspection")
            .expect("runner dropped history inspection reply")
    }

    async fn finish_runner_harness(harness: RunnerHarness) -> Agent<OpenAIConfig> {
        let RunnerHarness {
            control_tx,
            event_rx,
            task,
        } = harness;
        drop(control_tx);
        drop(event_rx);
        timeout(RUNNER_INTEGRATION_TIMEOUT, task)
            .await
            .expect("runner harness should stop")
            .expect("runner harness task should not panic")
    }

    fn records(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Vec<TranscriptRecord> {
        let recorder = transcript.lock().expect("lock transcript");
        read_records(recorder.path()).expect("read transcript")
    }

    fn project_terminal_runtime(events: &[RunnerEvent]) -> TuiRuntime {
        let mut projected = runtime();
        projected.runner_turn_active = true;
        projected.state_mut().phase = AppPhase::Running;
        for event in events {
            projected.apply_runner_event(event.clone());
        }
        projected
    }

    fn terminal_count(events: &[RunnerEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, RunnerEvent::Done | RunnerEvent::Interrupted))
            .count()
    }

    #[tokio::test]
    async fn runner_control_fifo_delegate_then_interrupt_before_first_poll_drops_unstarted_child() {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body("reusable child slot"),
        )])
        .await;
        let (sessions_dir, transcript) =
            test_transcript("fifo-delegate-before-interrupt", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let (mut harness, release) =
            start_paused_runner_harness(agent, Arc::clone(&transcript), sessions_dir.clone()).await;

        harness
            .send_command(RunnerCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "must not start".into(),
            })
            .expect("runner accepts delegate command");
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts delegate cancellation");
        release
            .send(())
            .expect("release the runner after both FIFO controls are queued");

        let interrupted_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(
            interrupted_events.last(),
            Some(RunnerEvent::Interrupted)
        ));
        assert_eq!(
            interrupted_events
                .iter()
                .filter(|event| matches!(event, RunnerEvent::Interrupted))
                .count(),
            1
        );
        assert_eq!(terminal_count(&interrupted_events), 1);
        assert!(
            !interrupted_events.iter().any(|event| matches!(
                event,
                RunnerEvent::Done | RunnerEvent::ChildAppEvent { .. }
            ))
        );
        assert!(matches!(
            server.requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        let pre_start_records = records(&transcript);
        assert!(
            !pre_start_records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::SubagentLifecycle { .. }))
        );
        assert!(!crate::transcript::child_sessions_dir(&sessions_dir).exists());

        harness
            .send_command(RunnerCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "prove the child slot is reusable".into(),
            })
            .expect("runner accepts follow-up delegate");
        server.expect_request(0).await;
        let follow_up_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(follow_up_events.last(), Some(RunnerEvent::Done)));
        assert_eq!(terminal_count(&follow_up_events), 1);

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_control_fifo_command_then_interrupt_before_first_poll_interrupts_prompt_and_runs_next_command()
     {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body("next prompt completed"),
        )])
        .await;
        let (sessions_dir, transcript) =
            test_transcript("fifo-prompt-before-interrupt", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let (mut harness, release) =
            start_paused_runner_harness(agent, Arc::clone(&transcript), sessions_dir).await;
        let mut dispatch_runtime = runtime();

        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
                "cancelled-before-start",
                crate::user_content::UserMessageContent::new(
                    "must not reach the provider",
                    Vec::new(),
                ),
            )),
            &harness.control_tx,
            true,
        );
        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::Interrupt,
            &harness.control_tx,
            true,
        );
        release
            .send(())
            .expect("release the runner after both FIFO controls are queued");

        let interrupted_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(
            interrupted_events.last(),
            Some(RunnerEvent::Interrupted)
        ));
        assert_eq!(terminal_count(&interrupted_events), 1);
        assert!(
            !interrupted_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Done))
        );
        assert!(matches!(
            server.requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
                "follow-up",
                crate::user_content::UserMessageContent::new(
                    "the next command still runs",
                    Vec::new(),
                ),
            )),
            &harness.control_tx,
            true,
        );
        server.expect_request(0).await;
        let follow_up_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(follow_up_events.last(), Some(RunnerEvent::Done)));
        assert!(
            !follow_up_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_control_fifo_prompt_then_interrupt_before_first_poll_does_not_interrupt_finalized_turn()
     {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body("this request must not run"),
        )])
        .await;
        let (sessions_dir, transcript) =
            test_transcript("fifo-prompt-finalized-history", Vec::new());
        {
            let mut recorder = transcript.lock().expect("lock transcript");
            recorder
                .record_user_message("completed request")
                .expect("record prior user message");
            recorder
                .record_turn_started(TurnStartedEvent {
                    turn_id: 41,
                    intent: "lightweight".into(),
                    directive: "complete the prior request".into(),
                    validation_reminder: "".into(),
                })
                .expect("record prior turn start");
            recorder
                .record_assistant_message("completed reply")
                .expect("record prior assistant message");
            recorder
                .record_turn_finalized(TurnFinalizedEvent {
                    turn_id: 41,
                    outcome: "completed".into(),
                    tool_call_count: 0,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                })
                .expect("record prior turn finalization");
        }
        let history_before = records(&transcript);
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let (mut harness, release) =
            start_paused_runner_harness(agent, Arc::clone(&transcript), sessions_dir).await;

        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "cancelled-before-start",
                crate::user_content::UserMessageContent::new(
                    "must not reach the provider",
                    Vec::new(),
                ),
            )))
            .expect("runner accepts prompt command");
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts prompt cancellation");
        release
            .send(())
            .expect("release the runner after both FIFO controls are queued");

        let interrupted_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(
            interrupted_events.last(),
            Some(RunnerEvent::Interrupted)
        ));
        assert_eq!(
            interrupted_events
                .iter()
                .filter(|event| matches!(event, RunnerEvent::Interrupted))
                .count(),
            1
        );
        assert_eq!(terminal_count(&interrupted_events), 1);
        assert!(matches!(
            server.requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        let history_after = records(&transcript);
        assert_eq!(history_after.len(), history_before.len());
        assert_eq!(
            history_after
                .iter()
                .filter(|record| matches!(record.event, TranscriptEvent::UserMessage { .. }))
                .count(),
            1
        );
        assert_eq!(
            history_after
                .iter()
                .filter(|record| matches!(record.event, TranscriptEvent::TurnStarted(_)))
                .count(),
            1
        );
        assert!(!history_after.iter().any(|record| matches!(
            record.event,
            TranscriptEvent::TurnInterrupted { turn_id: Some(41) }
        )));
        assert!(
            !history_after
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
        );

        let _ = finish_runner_harness(harness).await;
        server.abort().await;
    }

    #[tokio::test]
    async fn runner_control_fifo_interrupt_then_command_before_first_poll_discards_idle_interrupt()
    {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body("idle interrupt does not poison this prompt"),
        )])
        .await;
        let (sessions_dir, transcript) =
            test_transcript("fifo-interrupt-before-prompt", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let (mut harness, release) =
            start_paused_runner_harness(agent, Arc::clone(&transcript), sessions_dir).await;
        let mut dispatch_runtime = runtime();

        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::Interrupt,
            &harness.control_tx,
            true,
        );
        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::SubmitPrompt(UserMessageSubmission::new(
                "after-idle-interrupt",
                crate::user_content::UserMessageContent::new("this prompt must run", Vec::new()),
            )),
            &harness.control_tx,
            true,
        );
        release
            .send(())
            .expect("release the runner after both FIFO controls are queued");

        server.expect_request(0).await;
        let events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(events.last(), Some(RunnerEvent::Done)));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );
        assert_eq!(terminal_count(&events), 1);

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_control_fifo_command_then_interrupt_before_first_poll_cancels_manual_compaction_without_provider_request()
     {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body("summary that must not be requested"),
        )])
        .await;
        let (sessions_dir, transcript) = test_transcript(
            "fifo-compact-before-interrupt",
            vec![("older request".into(), "older reply".into())],
        );
        let mut agent = integration_agent(server.base_url.clone(), 32_000);
        rehydrate_agent_from_transcript(&mut agent, &transcript)
            .expect("seed compaction history from transcript");
        let (mut harness, release) =
            start_paused_runner_harness(agent, Arc::clone(&transcript), sessions_dir).await;
        let mut dispatch_runtime = runtime();

        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::Compact,
            &harness.control_tx,
            true,
        );
        command_dispatch::dispatch_command(
            &mut dispatch_runtime,
            RuntimeCommand::Interrupt,
            &harness.control_tx,
            true,
        );
        release
            .send(())
            .expect("release the runner after both FIFO controls are queued");

        let events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(events.last(), Some(RunnerEvent::Done)));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionFailed))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );
        assert_eq!(terminal_count(&events), 1);
        assert!(matches!(
            server.requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(
            !records(&transcript)
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
        );

        let _ = finish_runner_harness(harness).await;
        server.abort().await;
    }

    #[tokio::test]
    async fn runner_manual_compaction_cancel_before_persistence_rehydrates_and_drains_stale_cancel()
    {
        let mut server = spawn_controlled_sse_server(vec![
            ControlledSseResponse::Immediate(responses_sse_body("completed older turn")),
            ControlledSseResponse::Blocked(responses_sse_body("summary that must not persist")),
            ControlledSseResponse::Immediate(responses_sse_body("follow-up survives")),
        ])
        .await;
        let (sessions_dir, transcript) = test_transcript("manual-cancel", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "completed-turn",
                crate::user_content::UserMessageContent::new("complete an older turn", Vec::new()),
            )))
            .expect("runner accepts completed prompt");
        server.expect_request(0).await;
        let completed_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(completed_events.last(), Some(RunnerEvent::Done)));
        let durable_history = inspect_runner_history(&harness).await;
        assert!(
            records(&transcript)
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::TurnFinalized(_)))
        );

        harness
            .send_command(RunnerCommand::Compact)
            .expect("runner accepts manual compaction");
        server.expect_request(1).await;
        let (queued_history_tx, queued_history_rx) = oneshot::channel();
        harness
            .send_command(RunnerCommand::InspectHistory(queued_history_tx))
            .expect("runner queues a command behind manual compaction");
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts compaction cancellation");
        server.release.notify_one();
        let cancelled_events = runner_events_until_terminal(&mut harness).await;

        assert!(
            cancelled_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionStarted))
        );
        assert!(
            cancelled_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionFailed))
        );
        assert!(
            !cancelled_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
        );
        assert!(
            !cancelled_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );
        assert_eq!(terminal_count(&cancelled_events), 1);

        let projected = project_terminal_runtime(&cancelled_events);
        assert!(!projected.runner_turn_active);
        assert!(projected.state().pending_question.is_none());
        assert!(projected.state().pending_permission.is_none());
        assert_eq!(projected.state().phase, AppPhase::Completed);
        let queued_history = timeout(RUNNER_INTEGRATION_TIMEOUT, queued_history_rx)
            .await
            .expect("queued command is processed after manual compaction")
            .expect("runner keeps the queued command reply sender");
        assert_eq!(queued_history, durable_history);

        let durable_records = records(&transcript);
        assert!(
            !durable_records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
        );
        assert!(
            !durable_records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
        );

        // A second cancellation arrives while idle. The runner must consume it
        // before accepting the next operation rather than poisoning that prompt.
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts stale cancellation");
        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "follow-up",
                crate::user_content::UserMessageContent::new("follow up after compact", Vec::new()),
            )))
            .expect("runner accepts follow-up prompt");
        server.expect_request(2).await;
        let follow_up_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(follow_up_events.last(), Some(RunnerEvent::Done)));
        assert!(
            !follow_up_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );

        let agent = finish_runner_harness(harness).await;
        assert!(agent.history_for_test().iter().any(|item| {
            matches!(item, HistoryItem::AssistantText { text } if text == "follow-up survives")
        }));
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_manual_compaction_refreshes_session_token_usage_after_commit() {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body(&compaction_checkpoint("durable summary")),
        )])
        .await;
        // Short one-liners are comparable to the durable summary length, so seed
        // enough historical bulk that a successful compact must reduce tokens.
        let bulky_user = "older request ".repeat(120);
        let bulky_assistant = "older reply ".repeat(120);
        let (sessions_dir, transcript) = test_transcript(
            "manual-token-refresh",
            vec![
                (bulky_user.clone(), bulky_assistant.clone()),
                (bulky_user, bulky_assistant),
            ],
        );
        let mut agent = integration_agent(server.base_url.clone(), 32_000);
        rehydrate_agent_from_transcript(&mut agent, &transcript)
            .expect("seed agent from transcript");
        let before = manual_compaction_session_token_usage(&agent).expect("initial token usage");
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::Compact)
            .expect("runner accepts manual compaction");
        server.expect_request(0).await;
        let events = runner_events_until_terminal(&mut harness).await;

        let committed_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
            .expect("compaction committed event");
        let (usage_index, usage) = events
            .iter()
            .enumerate()
            .find_map(|(index, event)| match event {
                RunnerEvent::SessionTokenUsage(usage) => Some((index, usage)),
                _ => None,
            })
            .expect("session token usage event");
        let context_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::RuntimeContextUpdated(_)))
            .expect("runtime context event");
        let done_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::Done))
            .expect("done event");
        assert!(committed_index < usage_index);
        assert!(usage_index < context_index);
        assert!(context_index < done_index);
        assert!(usage.used_tokens < before.used_tokens);
        assert!(usage.input_tokens < before.input_tokens);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cached_tokens, 0);
        assert_eq!(usage.cache_report, None);

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_manual_compaction_persistence_wins_over_late_cancel() {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Immediate(
            responses_sse_body(&compaction_checkpoint("durable summary")),
        )])
        .await;
        let (sessions_dir, transcript) = test_transcript(
            "manual-persistence-wins",
            vec![("older request".into(), "older reply".into())],
        );
        let mut agent = integration_agent(server.base_url.clone(), 32_000);
        rehydrate_agent_from_transcript(&mut agent, &transcript)
            .expect("seed agent from transcript");
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::Compact)
            .expect("runner accepts manual compaction");
        server.expect_request(0).await;
        let mut events = runner_events_until_compaction_committed(&mut harness).await;

        assert!(
            records(&transcript)
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
        );
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts late cancellation");
        let committed_history = inspect_runner_history(&harness).await;
        events.extend(runner_events_until_terminal(&mut harness).await);

        assert!(
            events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
        );
        assert!(
            !events.iter().any(|event| matches!(
                event,
                RunnerEvent::CompactionFailed | RunnerEvent::Error(_)
            ))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );
        assert_eq!(terminal_count(&events), 1);
        let committed_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
            .expect("compaction committed event");
        let usage_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::SessionTokenUsage(_)))
            .expect("rehydrated token usage event");
        let context_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::RuntimeContextUpdated(_)))
            .expect("runtime context event");
        let done_index = events
            .iter()
            .position(|event| matches!(event, RunnerEvent::Done))
            .expect("done event");
        assert!(committed_index < usage_index);
        assert!(usage_index < context_index);
        assert!(context_index < done_index);
        assert!(
            committed_history
                .iter()
                .any(|item| matches!(item, HistoryItem::ContextSummary { .. }))
        );

        let mut restored = integration_agent(server.base_url.clone(), 32_000);
        rehydrate_agent_from_transcript(&mut restored, &transcript)
            .expect("rehydrate committed compaction");
        assert_eq!(restored.history_for_test(), committed_history.as_slice());

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_pressure_compaction_cancel_interrupts_enclosing_prompt_without_stale_cancel() {
        let mut server = spawn_controlled_sse_server(vec![
            ControlledSseResponse::Blocked(responses_sse_body("pressure summary")),
            ControlledSseResponse::Immediate(responses_sse_body("next prompt completed")),
        ])
        .await;
        let history = (0..24)
            .map(|index| {
                (
                    format!("older request {index}: {}", "x".repeat(1_000)),
                    format!("older reply {index}: {}", "y".repeat(1_000)),
                )
            })
            .collect();
        let (sessions_dir, transcript) = test_transcript("pressure-cancel", history);
        let mut agent = integration_agent(server.base_url.clone(), 8_000);
        rehydrate_agent_from_transcript(&mut agent, &transcript).expect("seed pressure history");
        agent.install_provider_usage_anchor_for_test(TokenUsageEstimate {
            used_tokens: 8_000,
            context_window_tokens: 8_000,
            input_tokens: 8_000,
            output_tokens: 0,
            cached_tokens: 0,
        });
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "pressure-prompt",
                crate::user_content::UserMessageContent::new("current pressure prompt", Vec::new()),
            )))
            .expect("runner accepts pressure prompt");
        server.expect_request(0).await;
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts prompt cancellation");
        server.release.notify_one();
        let interrupted_events = runner_events_until_terminal(&mut harness).await;

        assert!(
            interrupted_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionStarted))
        );
        assert!(matches!(
            interrupted_events.last(),
            Some(RunnerEvent::Interrupted)
        ));
        assert!(
            !interrupted_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::CompactionCommitted { .. }))
        );
        assert_eq!(terminal_count(&interrupted_events), 1);
        let interrupted_records = records(&transcript);
        assert!(
            !interrupted_records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::ContextCompaction(_)))
        );
        assert_eq!(
            interrupted_records
                .iter()
                .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
                .count(),
            1
        );

        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts stale cancellation");
        harness
            .send_command(RunnerCommand::SetModel("m2".into()))
            .expect("runner accepts model command");
        let _ = inspect_runner_history(&harness).await;
        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "post-pressure",
                crate::user_content::UserMessageContent::new(
                    "prompt after cancellation",
                    Vec::new(),
                ),
            )))
            .expect("runner accepts next prompt");
        server.expect_request(1).await;
        let follow_up_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(follow_up_events.last(), Some(RunnerEvent::Done)));
        assert!(
            !follow_up_events
                .iter()
                .any(|event| matches!(event, RunnerEvent::Interrupted))
        );

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_delegate_cancel_prioritizes_interrupt_and_reuses_child_slot() {
        let (race_control_tx, mut race_control_rx) = mpsc::unbounded_channel();
        let mut deferred_commands = VecDeque::new();
        race_control_tx
            .send(RunnerControl::Interrupt(test_interrupt()))
            .expect("queue simultaneous cancellation");
        let ready_delegate = std::future::ready(());
        tokio::pin!(ready_delegate);
        assert!(matches!(
            select_active_runner_operation(
                &mut race_control_rx,
                &mut deferred_commands,
                ready_delegate.as_mut(),
            )
            .await,
            ActiveRunnerOperation::Interrupted(_)
        ));

        let mut server = spawn_controlled_sse_server(vec![
            ControlledSseResponse::Blocked(responses_sse_body("cancelled child response")),
            ControlledSseResponse::Immediate(responses_sse_body("second child response")),
        ])
        .await;
        let (sessions_dir, transcript) = test_transcript("delegate-cancel", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "wait for cancellation".into(),
            })
            .expect("runner accepts first delegate");
        server.expect_request(0).await;
        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts delegate cancellation");
        server.release.notify_one();
        let cancelled_events = runner_events_until_terminal(&mut harness).await;
        assert_eq!(
            cancelled_events
                .iter()
                .filter(|event| matches!(event, RunnerEvent::Interrupted))
                .count(),
            1
        );
        assert_eq!(terminal_count(&cancelled_events), 1);

        harness
            .send_command(RunnerCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "prove the slot is reusable".into(),
            })
            .expect("runner accepts second delegate");
        server.expect_request(1).await;
        let second_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(second_events.last(), Some(RunnerEvent::Done)));
        assert_eq!(terminal_count(&second_events), 1);

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[tokio::test]
    async fn runner_interrupt_records_the_unmatched_started_turn() {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
            responses_sse_body("cancelled response"),
        )])
        .await;
        let (sessions_dir, transcript) = test_transcript("started-turn-interrupt", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);

        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "started-prompt",
                crate::user_content::UserMessageContent::new("wait for interruption", Vec::new()),
            )))
            .expect("runner accepts prompt");
        server.expect_request(0).await;
        let started_turn_id = records(&transcript)
            .iter()
            .find_map(|record| match &record.event {
                TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
                _ => None,
            })
            .expect("provider request follows a recorded turn start");

        harness
            .send_interrupt(test_interrupt())
            .expect("runner accepts prompt cancellation");
        let interrupted_events = runner_events_until_terminal(&mut harness).await;
        assert!(matches!(
            interrupted_events.last(),
            Some(RunnerEvent::Interrupted)
        ));
        assert_eq!(
            interrupted_events
                .iter()
                .filter(|event| matches!(event, RunnerEvent::Interrupted))
                .count(),
            1
        );
        assert_eq!(terminal_count(&interrupted_events), 1);

        let interrupted_turn_ids = records(&transcript)
            .iter()
            .filter_map(|record| match &record.event {
                TranscriptEvent::TurnInterrupted {
                    turn_id: Some(turn_id),
                } => Some(*turn_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(interrupted_turn_ids, vec![started_turn_id]);

        server.release.notify_one();
        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[test]
    fn session_title_base_sequence_remains_resolvable_for_interrupt_branch_scope() {
        let (_, transcript) = test_transcript("title-base-sequence-interrupt", Vec::new());
        let mut recorder = transcript.lock().expect("lock transcript");
        let title_sequence = read_records(recorder.path())
            .expect("read transcript records")
            .into_iter()
            .find(|record| matches!(record.event, TranscriptEvent::SessionTitle { .. }))
            .expect("session title exists")
            .sequence;

        recorder
            .record_context_branch_created("branch-a", ROOT_CONTEXT_BRANCH_ID, title_sequence, None)
            .expect("title sequence resolves on root branch");
    }

    #[test]
    fn record_interrupt_transcript_scopes_active_turn_to_recorder_branch() {
        let (_, transcript) = test_transcript("branch-scoped-interrupt", Vec::new());
        {
            let mut recorder = transcript.lock().expect("lock transcript");
            let root_leaf_sequence = read_records(recorder.path())
                .expect("read root records")
                .last()
                .expect("session metadata exists")
                .sequence;
            recorder
                .record_context_branch_created(
                    "branch-a",
                    ROOT_CONTEXT_BRANCH_ID,
                    root_leaf_sequence,
                    None,
                )
                .expect("create branch A");
            recorder
                .record_context_checkout("branch-a", root_leaf_sequence)
                .expect("checkout branch A");
            recorder.set_current_context_branch_id(Some("branch-a".into()));
            recorder
                .record_turn_started(turn_started(41))
                .expect("start branch A turn");

            recorder
                .record_context_branch_created(
                    "branch-b",
                    ROOT_CONTEXT_BRANCH_ID,
                    root_leaf_sequence,
                    None,
                )
                .expect("create branch B");
            recorder
                .record_context_checkout("branch-b", root_leaf_sequence)
                .expect("checkout branch B");
            recorder.set_current_context_branch_id(Some("branch-b".into()));
            recorder
                .record_turn_started(turn_started(42))
                .expect("start branch B turn");

            recorder.set_current_context_branch_id(Some("branch-a".into()));
        }

        record_interrupt_transcript(
            &transcript,
            &InterruptRequest {
                parent_tool_calls: vec![("call-a".into(), "shell__exec".into())],
                visible_child_session_id: None,
            },
        );

        let after = records(&transcript);
        let interruptions = after
            .iter()
            .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
            .collect::<Vec<_>>();
        assert_eq!(interruptions.len(), 1);
        assert!(matches!(
            &interruptions[0].event,
            TranscriptEvent::TurnInterrupted { turn_id: Some(41) }
        ));
        assert_eq!(
            interruptions[0].context_branch_id.as_deref(),
            Some("branch-a")
        );
        assert!(after.iter().any(|record| {
            matches!(
                &record.event,
                TranscriptEvent::ToolCallCancelled { call_id, name }
                    if call_id == "call-a" && name == "shell__exec"
            ) && record.context_branch_id.as_deref() == Some("branch-a")
        }));
    }

    #[test]
    fn record_interrupt_transcript_pre_start_cancellation_does_not_interrupt_sibling_turn() {
        let (_, transcript) = test_transcript("branch-pre-start-cancel", Vec::new());
        {
            let mut recorder = transcript.lock().expect("lock transcript");
            let root_leaf_sequence = read_records(recorder.path())
                .expect("read root records")
                .last()
                .expect("session metadata exists")
                .sequence;
            recorder
                .record_context_branch_created(
                    "branch-a",
                    ROOT_CONTEXT_BRANCH_ID,
                    root_leaf_sequence,
                    None,
                )
                .expect("create branch A");
            recorder
                .record_context_checkout("branch-a", root_leaf_sequence)
                .expect("checkout branch A");

            recorder
                .record_context_branch_created(
                    "branch-b",
                    ROOT_CONTEXT_BRANCH_ID,
                    root_leaf_sequence,
                    None,
                )
                .expect("create branch B");
            recorder
                .record_context_checkout("branch-b", root_leaf_sequence)
                .expect("checkout branch B");
            recorder.set_current_context_branch_id(Some("branch-b".into()));
            recorder
                .record_turn_started(turn_started(52))
                .expect("start branch B turn");

            recorder.set_current_context_branch_id(Some("branch-a".into()));
        }

        record_interrupt_transcript(
            &transcript,
            &InterruptRequest {
                parent_tool_calls: vec![("call-a".into(), "shell__exec".into())],
                visible_child_session_id: None,
            },
        );

        let after = records(&transcript);
        assert!(
            !after
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
        );
        assert!(after.iter().any(|record| {
            matches!(
                &record.event,
                TranscriptEvent::ToolCallCancelled { call_id, name }
                    if call_id == "call-a" && name == "shell__exec"
            ) && record.context_branch_id.as_deref() == Some("branch-a")
        }));
    }

    #[test]
    fn record_interrupt_transcript_normalizes_root_branch() {
        let (_, transcript) = test_transcript("root-active-turn-interrupt", Vec::new());
        {
            let mut recorder = transcript.lock().expect("lock transcript");
            assert_eq!(recorder.current_context_branch_id(), None);
            recorder
                .record_turn_started(turn_started(61))
                .expect("start root turn");
        }

        record_interrupt_transcript(&transcript, &test_interrupt());

        let interruptions = records(&transcript)
            .into_iter()
            .filter(|record| matches!(record.event, TranscriptEvent::TurnInterrupted { .. }))
            .collect::<Vec<_>>();
        assert_eq!(interruptions.len(), 1);
        assert!(matches!(
            &interruptions[0].event,
            TranscriptEvent::TurnInterrupted { turn_id: Some(61) }
        ));
        assert_eq!(interruptions[0].context_branch_id, None);
    }

    #[test]
    fn record_interrupt_transcript_fails_closed_when_branch_projection_cannot_resolve() {
        let (_, transcript) = test_transcript("unresolvable-branch-interrupt", Vec::new());
        {
            let mut recorder = transcript.lock().expect("lock transcript");
            recorder
                .record_turn_started(turn_started(71))
                .expect("start root turn");
            recorder.set_current_context_branch_id(Some("missing-branch".into()));
        }

        let before = serde_json::to_value(records(&transcript)).expect("serialize transcript");

        record_interrupt_transcript(
            &transcript,
            &InterruptRequest {
                parent_tool_calls: vec![("call-missing".into(), "shell__exec".into())],
                visible_child_session_id: None,
            },
        );

        let after = serde_json::to_value(records(&transcript)).expect("serialize transcript");
        assert_eq!(after, before);
    }

    #[tokio::test]
    async fn error_phase_double_escape_dispatches_to_a_live_runner_control_stream() {
        let mut server = spawn_controlled_sse_server(vec![ControlledSseResponse::Blocked(
            responses_sse_body("blocked normal prompt"),
        )])
        .await;
        let (sessions_dir, transcript) = test_transcript("error-phase-escape", Vec::new());
        let agent = integration_agent(server.base_url.clone(), 32_000);
        let mut harness = start_runner_harness(agent, Arc::clone(&transcript), sessions_dir);
        harness
            .send_command(RunnerCommand::Prompt(UserMessageSubmission::new(
                "blocked-prompt",
                crate::user_content::UserMessageContent::new("hold this prompt", Vec::new()),
            )))
            .expect("runner accepts blocked prompt");
        server.expect_request(0).await;

        let (_event_tx, event_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            event_rx,
            vec![AvailableModel::new("m1", "M1")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Error;
        let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);

        let first = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("first escape is accepted");
        assert_eq!(first, None);
        let second = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("second escape is accepted")
            .expect("second escape requests interruption");
        assert!(matches!(second, RuntimeCommand::Interrupt));
        command_dispatch::dispatch_command(&mut runtime, second, &harness.control_tx, true);

        server.release.notify_one();
        let events = runner_events_until_terminal(&mut harness).await;
        for event in &events {
            runtime.apply_runner_event(event.clone());
        }
        assert!(matches!(events.last(), Some(RunnerEvent::Interrupted)));
        assert!(!runtime.runner_turn_active);
        assert_eq!(runtime.state().phase, AppPhase::Completed);

        let _ = finish_runner_harness(harness).await;
        server.finish().await;
    }

    #[test]
    fn assistant_delta_buffer_aggregates_same_stream_across_frames() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new("hel")))
            .expect("send first delta");
        runtime.try_drain_runner_events();
        assert!(runtime.state().timeline.items().is_empty());

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new("lo")))
            .expect("send second delta");
        runner_tx
            .send(RunnerEvent::AssistantDone { message_id: None })
            .expect("send assistant done");
        runtime.try_drain_runner_events();

        assert!(matches!(
            runtime.state().timeline.items(),
            [TimelineItem::Assistant(message)] if message.text == "hello" && !message.streaming
        ));
    }

    #[test]
    fn assistant_delta_buffer_aggregates_child_stream_across_frames() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.state_mut().replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        for delta in ["hel", "lo"] {
            runner_tx
                .send(RunnerEvent::ChildAppEvent {
                    child_session_id: "child-session".into(),
                    agent_name: Some("explorer".into()),
                    parent_tool_call_id: Some("parent-call".into()),
                    event: AppEvent::AssistantDelta(AssistantDeltaEvent::new(delta)),
                })
                .expect("send child delta");
            runtime.try_drain_runner_events();
        }
        assert!(runtime.state().active_timeline().items().is_empty());

        runner_tx
            .send(RunnerEvent::ChildAppEvent {
                child_session_id: "child-session".into(),
                agent_name: Some("explorer".into()),
                parent_tool_call_id: Some("parent-call".into()),
                event: AppEvent::AssistantDone { message_id: None },
            })
            .expect("send child assistant done");
        runtime.try_drain_runner_events();

        assert!(matches!(
            runtime.state().active_timeline().items(),
            [TimelineItem::Assistant(message)] if message.text == "hello" && !message.streaming
        ));
    }

    #[test]
    fn assistant_delta_buffer_commits_through_last_newline_and_retains_tail() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(
                "one\ntwo",
            )))
            .expect("send delta");
        runtime.try_drain_runner_events();
        assert!(matches!(
            runtime.state().timeline.items(),
            [TimelineItem::Assistant(message)] if message.text == "one\n"
        ));

        runner_tx
            .send(RunnerEvent::AssistantDone { message_id: None })
            .expect("send assistant done");
        runtime.try_drain_runner_events();
        assert!(matches!(
            runtime.state().timeline.items(),
            [TimelineItem::Assistant(message)] if message.text == "one\ntwo" && !message.streaming
        ));
    }

    #[test]
    fn assistant_delta_buffer_flushes_after_wait_threshold() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(
                "waiting",
            )))
            .expect("send delta");
        runtime.try_drain_runner_events();
        runtime
            .assistant_delta_buffer
            .as_mut()
            .expect("buffered delta")
            .started_at -= ASSISTANT_DELTA_BUFFER_MAX_WAIT;
        runtime.try_drain_runner_events();

        assert!(matches!(
            runtime.state().timeline.items(),
            [TimelineItem::Assistant(message)] if message.text == "waiting"
        ));
    }

    #[test]
    fn assistant_delta_buffer_flushes_at_byte_threshold() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(
                "x".repeat(ASSISTANT_DELTA_BUFFER_MAX_BYTES),
            )))
            .expect("send threshold delta");
        runtime.try_drain_runner_events();

        assert!(matches!(
            runtime.state().timeline.items(),
            [TimelineItem::Assistant(message)] if message.text.len() == ASSISTANT_DELTA_BUFFER_MAX_BYTES
        ));
    }

    #[test]
    fn assistant_delta_buffer_flushes_before_different_stream_and_tool_events() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );

        runner_tx
            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(
                "parent",
            )))
            .expect("send parent delta");
        runner_tx
            .send(RunnerEvent::ChildAppEvent {
                child_session_id: "child-1".into(),
                agent_name: Some("explorer".into()),
                parent_tool_call_id: Some("parent-call".into()),
                event: AppEvent::AssistantDelta(AssistantDeltaEvent::new("child")),
            })
            .expect("send child delta");
        runner_tx
            .send(RunnerEvent::ToolStarted(ToolStartedEvent::new(
                "call-1",
                "shell__exec",
                "run",
            )))
            .expect("send tool event");
        runner_tx
            .send(RunnerEvent::Done)
            .expect("send terminal event");
        runtime.try_drain_runner_events();

        assert!(matches!(
            runtime.state().timeline.items(),
            [
                TimelineItem::Assistant(parent),
                TimelineItem::Tool(_),
            ] if parent.text == "parent" && !parent.streaming
        ));
        assert!(
            runtime.assistant_delta_buffer.is_none(),
            "the child delta commits before the following tool event"
        );
    }

    #[test]
    fn bounded_runner_event_drain_keeps_double_escape_cancel_dispatch_fair() {
        let (runner_tx, runner_rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::default(),
            runner_rx,
            vec![AvailableModel::new("m1", "M1")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        runtime.runner_turn_active = true;
        runtime.state_mut().phase = AppPhase::Error;
        for index in 0..512 {
            runner_tx
                .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(
                    format!("flood-{index}"),
                )))
                .expect("queue runner flood event");
        }

        runtime.try_drain_runner_events();
        assert!(runtime.runner_rx.try_recv().is_ok(), "drain stays bounded");

        let (control_tx, mut control_rx) = mpsc::unbounded_channel();
        let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(
            runtime
                .handle_input_action(map_key_event(runtime.state(), escape))
                .expect("first escape is accepted"),
            None
        );
        let command = runtime
            .handle_input_action(map_key_event(runtime.state(), escape))
            .expect("second escape is accepted")
            .expect("second escape requests interruption");
        command_dispatch::dispatch_command(&mut runtime, command, &control_tx, true);
        assert!(
            matches!(control_rx.try_recv(), Ok(RunnerControl::Interrupt(_))),
            "interrupt dispatch makes progress after flood"
        );
    }
}
