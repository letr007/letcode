use std::collections::VecDeque;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent, ManualCompactionOutcome, SubagentInvocation};
use crate::command::{
    ChildNavigation as SharedChildNavigation, CommandIntent, ToolOutputMode,
    TranscriptScrollbarMode, help_summary, parse_command,
};
use crate::mcp;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::runtime_context::RuntimeActiveContext;
use crate::subagent::SubagentRuntime;
use crate::tool::{ToolHandler, normalize_subagent_input};
use crate::transcript::{
    SessionSummary, TranscriptRecorder, has_session_content, list_child_sessions_for_parent,
    list_sessions, read_child_session_records_allow_partial_tail, read_records,
    remove_empty_session_file, restore_max_turn_id, sort_child_session_summaries,
    sync_recorder_branch, transcript_projection,
};
use crate::user_content::{UserImageAttachment, UserMessageSubmission};

use super::events::{
    AppEvent, AssistantDeltaEvent, ErrorEvent, NoticeEvent, RuntimeContextDisposition,
    RuntimeContextUpdatedEvent, TokenUsageEvent,
};
use super::input::{
    InputAction, apply_edit_action, map_key_event, map_mouse_event, map_paste_event,
};
use super::preferences::TuiPreferences;
use super::render;
use super::runner::{AgentRunner, RunnerEvent, RunnerPermissionRequest, RunnerQuestionRequest};
use super::slash::{SlashCommandEntry, matching_completion_commands};
use super::state::{
    ContextDetailTarget, DialogItem, DialogKind, DialogState, PendingQuestionState,
    QuestionAdvance, ToastKind, TuiState,
};
use super::terminal::OwnedTerminal;
use super::timeline::{COMPACTION_MESSAGE_ID, COMPACTION_SEPARATOR_LABEL, compaction_separator};
#[path = "runtime/command_dispatch.rs"]
mod command_dispatch;
#[path = "runtime/lifecycle.rs"]
mod lifecycle;
#[path = "runtime/permission_lifecycle.rs"]
mod permission_lifecycle;
#[path = "runtime/queued_prompt.rs"]
mod queued_prompt;
use async_openai::config::Config;
use lifecycle::{active_turn_state, build_interrupt_request, has_active_or_pending_runner_turn};
use permission_lifecycle::PermissionLifecycleController;
use queued_prompt::{QueuedPromptDoneDisposition, QueuedPromptLifecycle};
use serde_json::json;
use std::sync::atomic::AtomicU64;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, Ordering},
};

const PAGE_SCROLL_ROWS: u16 = 10;
const CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS: u8 = 20;
const TUI_FRAME_POLL_INTERVAL: Duration = Duration::from_millis(33);
static NEXT_SUBMISSION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ATTACHMENT_ID: AtomicU64 = AtomicU64::new(1);

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct InterruptRequest {
    parent_tool_calls: Vec<(String, String)>,
    visible_child_session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableModel {
    pub id: String,
    pub label: String,
    pub context_window_tokens: Option<u64>,
    pub reasoning_effort: Option<ModelReasoningEffort>,
}

impl AvailableModel {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            context_window_tokens: None,
            reasoning_effort: None,
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
        }
    }

    pub fn with_context_window_and_reasoning(
        id: impl Into<String>,
        label: impl Into<String>,
        context_window_tokens: Option<u64>,
        reasoning_effort: Option<ModelReasoningEffort>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            context_window_tokens,
            reasoning_effort,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCommand {
    SubmitPrompt(UserMessageSubmission),
    DelegateSubagent { agent_name: String, task: String },
    Compact,
    ShowBranchTree,
    ListBranches,
    CreateBranch { label: Option<String> },
    CheckoutBranch(String),
    ViewChild(ChildNavigation),
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    Interrupt,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildNavigation {
    First,
    Next,
    Prev,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SubmittedCommand {
    LocalOnly,
    Runtime(RuntimeCommand),
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
    history_draft: Option<String>,
    available_models: Vec<AvailableModel>,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
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
        }
    }

    pub fn state(&self) -> &TuiState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut TuiState {
        &mut self.state
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
        self.state.set_footer(
            "Question tool is waiting for your reply",
            Some("Pick an option, type your own answer, or press Esc to cancel".into()),
        );
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

    fn cancel_pending_question(&mut self, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        let handle = self.pending_question_handle.take();
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.cancel(reason.clone())
        {
            self.state.set_footer(
                "Question closed locally",
                Some(format!("failed to notify sender: {error}")),
            );
            return Err(error);
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
                .set_footer("Answer every question before confirming", None);
            return Ok(());
        }

        let Some(question) = self.state.pending_question.as_ref() else {
            return Err(anyhow!("no pending question"));
        };
        if question.has_invalid_single_response() {
            self.state
                .set_footer("Single-select questions accept only one answer", None);
            return Ok(());
        }

        let response = question.build_response();
        let handle = self.pending_question_handle.take();
        self.clear_pending_question();
        if let Some(handle) = handle
            && let Err(error) = handle.answer(response)
        {
            self.state.set_footer(
                "Question answered locally",
                Some(format!("failed to deliver answer: {error}")),
            );
            return Err(error);
        }
        self.state.set_footer("Question answered", None);
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
        while let Ok(event) = self.runner_rx.try_recv() {
            self.apply_runner_event(event);
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
                    self.state.set_footer(
                        "Question already pending",
                        Some("Resolve the current prompt first".into()),
                    );
                    suppress_app_event = true;
                }
            }
            RunnerEvent::PermissionRequested { event, handle } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state.set_footer(
                        "Question already pending",
                        Some("Resolve the current question before approving tools".into()),
                    );
                    suppress_app_event = true;
                } else if let Err(handle) = self
                    .permission_lifecycle
                    .begin_parent(event.clone(), handle.clone())
                {
                    let _ = handle.deny();
                    self.state.set_footer(
                        "Permission already pending",
                        Some("Resolve the current permission prompt first".into()),
                    );
                    suppress_app_event = true;
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
                    self.state.set_footer(
                        "Question already pending",
                        Some("Resolve the current prompt first".into()),
                    );
                    suppress_app_event = true;
                }
            }
            RunnerEvent::ChildPermissionRequested {
                child_session_id,
                event,
                handle,
            } => {
                if self.state.pending_question.is_some() {
                    let _ = handle.deny();
                    self.state.set_footer(
                        "Question already pending",
                        Some("Resolve the current question before approving tools".into()),
                    );
                } else if let Err(handle) = self.permission_lifecycle.begin_child(
                    child_session_id.clone(),
                    event.clone(),
                    handle.clone(),
                ) {
                    let _ = handle.deny();
                    self.state.set_footer(
                        "Permission already pending",
                        Some("Resolve the current permission prompt first".into()),
                    );
                } else {
                    self.state.apply_child_app_event(
                        child_session_id,
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
            RunnerEvent::QueuedPromptAccepted { prompt } => {
                self.queued_prompt_lifecycle.accept(&prompt.id);
            }
            RunnerEvent::Interrupted => {
                self.permission_lifecycle.clear_if_parent();
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
                self.state.set_footer(
                    "Session resumed",
                    Some(format!(
                        "{} ({} messages, {} evidence)",
                        session_id, message_count, evidence_count
                    )),
                );
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
                        runtime_context.clone(),
                    )
                {
                    self.state.show_toast(
                        format!("Context projection failed: {error}"),
                        ToastKind::Error,
                    );
                    return;
                }
                self.state.set_footer(
                    format!("Viewing {agent_name}"),
                    Some(format!(
                        "child {}/{} · {} · /parent to return",
                        index + 1,
                        total,
                        short_session_id(child_session_id)
                    )),
                );
            }
            RunnerEvent::ContextBranchesLoaded { branches } => {
                if let Err(error) = self.open_branch_dialog(branches) {
                    self.state
                        .set_footer("Failed to open context tree", Some(error.to_string()));
                }
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
                    .set_footer("New session started", Some(session_id.clone()));
            }
            RunnerEvent::Status(message) => {
                self.state.set_footer(message.clone(), None);
            }
            RunnerEvent::ChildAppEvent {
                child_session_id,
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
                self.state
                    .apply_child_app_event(child_session_id, event.clone());
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
                    Ok(Some(self.cycle_reasoning_effort_command()))
                }
            }
            InputAction::ChildPrefix => {
                self.state.child_navigation_prefix = true;
                self.state.child_navigation_prefix_ticks_remaining =
                    CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS;
                self.state.set_footer(
                    "Child navigation",
                    Some("Down enter child · Left/Right cycle · Up return".into()),
                );
                Ok(None)
            }
            InputAction::ChildFirst => Ok(Some(RuntimeCommand::ViewChild(ChildNavigation::First))),
            InputAction::ChildNext => Ok(Some(RuntimeCommand::ViewChild(ChildNavigation::Next))),
            InputAction::ChildPrev => Ok(Some(RuntimeCommand::ViewChild(ChildNavigation::Prev))),
            InputAction::ChildParent => {
                if self.state.is_read_only_child_view() {
                    self.state.restore_parent_timeline_view();
                    self.state.set_footer("Parent transcript", None);
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
                    Action::Advanced => {
                        self.state.set_footer(
                            "Review the next question",
                            Some("Use left and right to revisit any answer".into()),
                        );
                    }
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
                    self.state.set_footer("Custom answer closed", None);
                } else {
                    self.cancel_pending_question("question dismissed by user")?;
                    self.state.set_footer("Question dismissed", None);
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
            InputAction::DialogCancel => {
                let detail_focused = self.state.dialog().is_some_and(|dialog| {
                    dialog.kind == DialogKind::ContextPicker && dialog.detail_focused
                });
                if detail_focused {
                    if let Some(dialog) = self.state.dialog_mut() {
                        dialog.detail_focused = false;
                        dialog.detail_scroll = 0;
                    }
                    self.state
                        .set_footer("Context", Some("Search and open details".into()));
                } else {
                    self.state.close_dialog();
                    self.state.set_footer("Dialog closed", None);
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
                        self.state
                            .set_footer("Ready", Some("Enter a prompt or /help commands".into()));
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
        _terminal: &mut OwnedTerminal,
        drawer: &mut D,
    ) -> io::Result<()> {
        self.try_drain_runner_events();
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

    fn handle_submit(&mut self) -> Result<Option<RuntimeCommand>> {
        if self.state.pending_permission.is_some() || self.state.pending_question.is_some() {
            return Ok(None);
        }

        if self.state.slash_panel_is_open()
            && let Some(selected) = self.selected_slash_command()
        {
            let current = self.state.input_buffer.trim();
            if current != selected.command {
                self.state.set_input(selected.insert_text);
                return Ok(None);
            }
        }

        let mut content = self.state.composer_content();
        content.text = content.text.trim().to_string();
        if content.is_empty() {
            return Ok(None);
        }

        let prompt = content.text.clone();

        self.reset_history_navigation();

        let parsed_command = parse_command(&prompt);
        let active_runner_turn = self.has_active_or_pending_runner_turn();
        let running_navigation = active_runner_turn && child_view_allows_prompt(&prompt);
        if active_runner_turn && !running_navigation {
            if matches!(&parsed_command, Ok(CommandIntent::Delegate { .. })) {
                self.state.set_footer(
                    "Turn still running",
                    Some("Interrupt the current turn before delegating to an expert".into()),
                );
                return Ok(None);
            }

            if !prompt.starts_with('/')
                && !prompt.starts_with('@')
                && !self.state.is_read_only_child_view()
            {
                self.queue_prompt(UserMessageSubmission::new(next_submission_id(), content));
                return Ok(None);
            }

            self.state.set_footer(
                "Turn still running",
                Some("Press Esc twice to interrupt".into()),
            );
            return Ok(None);
        }

        if self.state.is_read_only_child_view() && !child_view_allows_prompt(&prompt) {
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
        self.state.clear_composer_attachments();
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        self.queued_prompt_lifecycle.clear_dispatch_ready();
        self.runner_turn_active = true;
        self.state.set_footer(
            "Submitting prompt",
            Some("Waiting for runner events".into()),
        );
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
                self.history_draft = Some(self.state.input_buffer.clone());
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

        let draft = self.history_draft.take().unwrap_or_default();
        self.history_selection = None;
        self.state.set_input(draft);
    }

    fn reset_history_navigation(&mut self) {
        self.history_selection = None;
        self.history_draft = None;
    }

    fn queue_prompt(&mut self, prompt: UserMessageSubmission) {
        self.state.clear_input();
        self.state.clear_composer_attachments();
        self.state.mark_session_active();
        self.submitted_prompts.push(prompt.content.text.clone());
        self.queued_prompts.push_back(prompt.clone());
        self.state.push_queued_user_message_preview(prompt);
        let queued = self.queued_prompts.len();
        self.state.set_footer(
            "Queued prompt",
            Some(format!("runs after current turn · {queued} queued")),
        );
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
        let remaining = self.queued_prompts.len().saturating_sub(1);
        self.state.set_footer(
            "Submitting queued prompt",
            Some(format!("{remaining} queued")),
        );
        Some(RuntimeCommand::SubmitPrompt(prompt))
    }

    fn handle_interrupt(&mut self) -> Result<Option<RuntimeCommand>> {
        if !self.has_active_or_pending_runner_turn() {
            self.interrupt_confirmation_pending = false;
            return Ok(None);
        }

        if !self.interrupt_confirmation_pending {
            self.interrupt_confirmation_pending = true;
            self.state.set_footer(
                "Press Esc again to interrupt",
                Some("Current assistant turn is still running".into()),
            );
            return Ok(None);
        }

        self.interrupt_confirmation_pending = false;
        self.state.set_footer(
            "Interrupting",
            Some("Stopping current assistant turn".into()),
        );
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
        match parsed {
            Ok(CommandIntent::Prompt(_)) => Ok(None),
            Ok(CommandIntent::Exit) => {
                self.state.apply_event(AppEvent::Quit);
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            Ok(CommandIntent::Help) => {
                self.push_command_notice(help_summary());
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            Ok(CommandIntent::ModelShow) => self.show_model_dialog(),
            Ok(CommandIntent::ModelSet(model_id)) => self.handle_model_selection(model_id),
            Ok(CommandIntent::ReasoningShow) => self.show_reasoning_dialog(),
            Ok(CommandIntent::ReasoningSet(effort)) => {
                Ok(Some(self.set_reasoning_effort_command(effort)))
            }
            Ok(CommandIntent::PermissionShow) => self.show_permission_dialog(),
            Ok(CommandIntent::PermissionSet(mode)) => {
                Ok(Some(self.set_permission_mode_command(mode)))
            }
            Ok(CommandIntent::ToolOutputSet(mode)) => self.handle_tool_output_command(mode),
            Ok(CommandIntent::TranscriptScrollbarSet(mode)) => {
                Ok(Some(self.handle_transcript_scrollbar_command(mode)))
            }
            Ok(CommandIntent::Compact) => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.runner_turn_active = true;
                self.state.set_footer("Compacting context", None);
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Compact)))
            }
            Ok(CommandIntent::Tree) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ShowBranchTree,
            ))),
            Ok(CommandIntent::Branches) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ListBranches,
            ))),
            Ok(CommandIntent::BranchCreate(label)) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::CreateBranch { label },
            ))),
            Ok(CommandIntent::CheckoutBranch(branch_id)) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::CheckoutBranch(branch_id),
            ))),
            Ok(CommandIntent::ResumeShow) => self.show_resume_dialog(),
            Ok(CommandIntent::Resume(session_id)) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ResumeSession(session_id),
            ))),
            Ok(CommandIntent::NewSession) => {
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::NewSession)))
            }
            Ok(CommandIntent::ContextBrowse) => self.show_context_dialog(),
            Ok(CommandIntent::Delegate { agent_name, task }) => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.runner_turn_active = true;
                self.state
                    .timeline
                    .push_delegation(agent_name.clone(), task.clone());
                self.state
                    .set_footer(format!("Starting {agent_name}"), Some(task.clone()));
                Ok(Some(SubmittedCommand::Runtime(
                    RuntimeCommand::DelegateSubagent { agent_name, task },
                )))
            }
            Ok(CommandIntent::Child(navigation)) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ViewChild(map_child_navigation(navigation)),
            ))),
            Ok(CommandIntent::Parent) => {
                if self.state.transcript_view.is_child() {
                    self.state.restore_parent_timeline_view();
                    self.state.set_footer("Parent transcript", None);
                    Ok(Some(SubmittedCommand::LocalOnly))
                } else {
                    Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::ViewParent)))
                }
            }
            Err(error) => {
                self.push_command_notice(error.message());
                Ok(Some(SubmittedCommand::LocalOnly))
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
            self.state.set_footer(
                "Tool output mode changed",
                Some(format!("{} · save failed: {}", mode.label(), error)),
            );
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        self.state
            .set_footer("Tool output mode changed", Some(mode.label().to_string()));
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
            self.state.set_footer(
                "Transcript scrollbar",
                Some(format!(
                    "{} · save failed: {}",
                    if visible { "visible" } else { "hidden" },
                    error
                )),
            );
            return SubmittedCommand::LocalOnly;
        }
        self.state.set_footer(
            "Transcript scrollbar",
            Some(if visible { "visible" } else { "hidden" }.into()),
        );
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
        self.state.set_footer(
            "Permission dialog",
            Some("Choose a mode and press Enter".into()),
        );
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
        self.state.set_footer(
            "Model dialog",
            Some("Choose a model and press Enter".into()),
        );
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

        self.state.set_model(model.id.clone(), model.label.clone());
        self.state
            .set_model_context_window(model.context_window_tokens);
        self.state
            .set_reasoning_effort_label(Some(reasoning_effort_status_label(
                model.reasoning_effort,
            )));
        self.state.set_footer(
            "Model updated",
            Some(format!("using {} ({})", model.label, model.id)),
        );
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
        self.state.set_footer(
            "Session dialog",
            Some("Choose a session and press Enter".into()),
        );
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn open_branch_dialog(
        &mut self,
        branches: &[transcript_projection::ContextBranchInfo],
    ) -> Result<Option<SubmittedCommand>> {
        if branches.is_empty() {
            self.push_command_notice("No context branches found");
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        let current_branch = branches
            .iter()
            .find(|branch| branch.is_current)
            .map(|branch| branch.branch_id.clone());
        let items = branch_dialog_items(branches);
        let mut dialog = DialogState::new(
            DialogKind::BranchPicker,
            "Context tree",
            Some("Choose a branch and press Enter".into()),
            items,
        );
        if let Some(current_branch) = current_branch
            && let Some(index) = dialog
                .items
                .iter()
                .position(|item| item.id == current_branch)
        {
            dialog.selected = index;
        }
        self.state.open_dialog(dialog);
        self.state.set_footer(
            "Context tree",
            Some("Search, move, and press Enter to checkout a branch".into()),
        );
        Ok(Some(SubmittedCommand::LocalOnly))
    }

    fn show_reasoning_dialog(&mut self) -> Result<Option<SubmittedCommand>> {
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            "Reasoning effort",
            Some("Select how much reasoning the model should use".into()),
            reasoning_dialog_items(),
        );
        dialog.selected = reasoning_dialog_selected_index(self.current_reasoning_effort());
        self.state.open_dialog(dialog);
        self.state.set_footer(
            "Reasoning dialog",
            Some("Choose an effort and press Enter".into()),
        );
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
        self.state
            .set_footer("Context", Some("Search and open details".into()));
        Ok(Some(SubmittedCommand::LocalOnly))
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
                    .and_then(|model| model.reasoning_effort);
                self.state
                    .set_reasoning_effort_label(Some(reasoning_effort_status_label(
                        reasoning_effort,
                    )));
                self.state.set_footer(
                    "Model updated",
                    Some(format!("using {} ({})", selected.label, selected.id)),
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
                self.state.set_footer(
                    "Permission mode updated",
                    Some(format!("mode is now {label}")),
                );
                Ok(Some(RuntimeCommand::SetPermissionMode(mode)))
            }
            DialogKind::ReasoningPicker => {
                self.state.close_dialog();
                let effort = parse_reasoning_effort(&selected.id)
                    .expect("reasoning picker items should use valid effort ids");
                self.state
                    .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(effort))));
                Ok(Some(RuntimeCommand::SetReasoningEffort(effort)))
            }
            DialogKind::SessionPicker => {
                self.state.close_dialog();
                Ok(Some(RuntimeCommand::ResumeSession(selected.id)))
            }
            DialogKind::BranchPicker => {
                self.state.close_dialog();
                Ok(Some(RuntimeCommand::CheckoutBranch(selected.id)))
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
            DialogKind::ContextDetail => {
                self.state.close_dialog();
                Ok(None)
            }
        }
    }

    fn notify_context_dialog_issue(&mut self, summary: &str, detail: &str) {
        self.state.set_footer(summary, Some(detail.into()));
        self.show_toast(summary, ToastKind::Error);
        self.state
            .push_active_notice(format!("{summary} · {detail}"));
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
        self.state.set_footer(
            "Permission mode updated",
            Some(format!("mode is now {label}")),
        );
        SubmittedCommand::Runtime(RuntimeCommand::SetPermissionMode(mode))
    }

    fn set_reasoning_effort_command(&mut self, effort: ModelReasoningEffort) -> SubmittedCommand {
        self.state
            .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(effort))));
        SubmittedCommand::Runtime(RuntimeCommand::SetReasoningEffort(effort))
    }

    fn cycle_reasoning_effort_command(&mut self) -> RuntimeCommand {
        let next = next_reasoning_effort(self.current_reasoning_effort());
        self.state
            .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(next))));
        RuntimeCommand::SetReasoningEffort(next)
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
        self.state.set_footer(message.into(), None);
    }

    fn push_child_view_read_only_notice(&mut self) {
        self.state.set_footer(
            "Viewing child transcript",
            Some("Use /parent to return before changing the parent session".into()),
        );
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
    ShowBranchTree,
    ListBranches,
    CreateBranch {
        label: Option<String>,
    },
    CheckoutBranch(String),
    ViewChild {
        navigation: ChildNavigation,
        anchor_child_session_id: Option<String>,
    },
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
}

fn parse_reasoning_effort(value: &str) -> Option<ModelReasoningEffort> {
    crate::command::parse_reasoning_effort(value)
}

fn reasoning_effort_config_label(effort: ModelReasoningEffort) -> &'static str {
    match effort {
        ModelReasoningEffort::None => "none",
        ModelReasoningEffort::Minimal => "minimal",
        ModelReasoningEffort::Low => "low",
        ModelReasoningEffort::Medium => "medium",
        ModelReasoningEffort::High => "high",
        ModelReasoningEffort::Xhigh => "xhigh",
    }
}

fn reasoning_effort_status_label(effort: Option<ModelReasoningEffort>) -> String {
    match effort {
        Some(ModelReasoningEffort::None) | None => "off".into(),
        Some(effort) => reasoning_effort_config_label(effort).into(),
    }
}

fn next_reasoning_effort(current: Option<ModelReasoningEffort>) -> ModelReasoningEffort {
    match current {
        None => ModelReasoningEffort::Minimal,
        Some(ModelReasoningEffort::None) => ModelReasoningEffort::Minimal,
        Some(ModelReasoningEffort::Minimal) => ModelReasoningEffort::Low,
        Some(ModelReasoningEffort::Low) => ModelReasoningEffort::Medium,
        Some(ModelReasoningEffort::Medium) => ModelReasoningEffort::High,
        Some(ModelReasoningEffort::High) => ModelReasoningEffort::Xhigh,
        Some(ModelReasoningEffort::Xhigh) => ModelReasoningEffort::None,
    }
}

fn reasoning_dialog_items() -> Vec<DialogItem> {
    vec![
        DialogItem::new("none", "Off", Some("Do not request extra reasoning".into())),
        DialogItem::new(
            "minimal",
            "Minimal",
            Some("Smallest reasoning budget".into()),
        ),
        DialogItem::new("low", "Low", Some("Light reasoning budget".into())),
        DialogItem::new("medium", "Medium", Some("Balanced reasoning budget".into())),
        DialogItem::new("high", "High", Some("Deeper reasoning budget".into())),
        DialogItem::new("xhigh", "XHigh", Some("Maximum reasoning budget".into())),
    ]
}

fn reasoning_dialog_selected_index(current: Option<ModelReasoningEffort>) -> usize {
    match current {
        None | Some(ModelReasoningEffort::None) => 0,
        Some(ModelReasoningEffort::Minimal) => 1,
        Some(ModelReasoningEffort::Low) => 2,
        Some(ModelReasoningEffort::Medium) => 3,
        Some(ModelReasoningEffort::High) => 4,
        Some(ModelReasoningEffort::Xhigh) => 5,
    }
}

fn session_dialog_item(session: &SessionSummary) -> DialogItem {
    let label = session
        .title
        .clone()
        .or_else(|| session.last_user_summary.clone())
        .or_else(|| session.last_assistant_summary.clone())
        .unwrap_or_else(|| "empty session".into());
    let timestamp_ms = session.last_timestamp_ms.or(session.first_timestamp_ms);
    let section = timestamp_ms
        .map(session_section_label)
        .unwrap_or_else(|| "Unknown date".into());
    let right_detail = timestamp_ms
        .map(session_time_label)
        .unwrap_or_else(|| "--:--".into());

    DialogItem::new(
        session.session_id.clone(),
        label,
        Some(session.session_id.clone()),
    )
    .with_section(section)
    .with_right_detail(right_detail)
}

fn session_section_label(timestamp_ms: u128) -> String {
    let (year, month, day) = utc_date_parts(timestamp_ms);
    let today = utc_date_parts(unix_timestamp_ms_for_tui());
    if (year, month, day) == today {
        return "Today".into();
    }

    let weekday = weekday_name(year, month, day);
    let month = month_name(month);
    format!("{weekday} {month} {day:02} {year}")
}

fn session_time_label(timestamp_ms: u128) -> String {
    let total_seconds = (timestamp_ms / 1_000) as u64;
    let seconds_in_day = total_seconds % 86_400;
    let hour = seconds_in_day / 3_600;
    let minute = (seconds_in_day % 3_600) / 60;
    let suffix = if hour < 12 { "AM" } else { "PM" };
    let display_hour = match hour % 12 {
        0 => 12,
        hour => hour,
    };
    format!("{display_hour}:{minute:02} {suffix}")
}

fn utc_date_parts(timestamp_ms: u128) -> (i32, u32, u32) {
    let days = (timestamp_ms / 1_000 / 86_400) as i64;
    civil_from_days(days)
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

fn weekday_name(year: i32, month: u32, day: u32) -> &'static str {
    let mut month = month as i32;
    let mut year = year;
    if month < 3 {
        month += 12;
        year -= 1;
    }
    let k = year % 100;
    let j = year / 100;
    let h = (day as i32 + (13 * (month + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    match h {
        0 => "Sat",
        1 => "Sun",
        2 => "Mon",
        3 => "Tue",
        4 => "Wed",
        5 => "Thu",
        _ => "Fri",
    }
}

fn month_name(month: u32) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        _ => "Dec",
    }
}

fn unix_timestamp_ms_for_tui() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn empty_session_path(path: &std::path::Path) -> Option<PathBuf> {
    let records = crate::transcript::read_records(path).ok()?;
    (!has_session_content(&records)).then(|| path.to_path_buf())
}

fn remove_current_empty_session(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Result<bool> {
    let path = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();

    remove_empty_session_file(path)
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

fn map_child_navigation(navigation: SharedChildNavigation) -> ChildNavigation {
    match navigation {
        SharedChildNavigation::First => ChildNavigation::First,
        SharedChildNavigation::Next => ChildNavigation::Next,
        SharedChildNavigation::Prev => ChildNavigation::Prev,
    }
}

fn current_session_records(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<(String, Vec<crate::transcript::TranscriptRecord>)> {
    let (session_id, path) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            recorder.path().to_path_buf(),
        )
    };
    let records = crate::transcript::read_records(path)?;
    Ok((session_id, records))
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
        None,
        transcript_projection::SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
    )
}

fn runtime_context_from_records(
    records: &[crate::transcript::TranscriptRecord],
    session_id: &str,
    branch_id: Option<&str>,
) -> Result<RuntimeActiveContext> {
    let snapshot = transcript_projection::project_runtime_restore_snapshot(
        session_id.to_string(),
        records.to_vec(),
        None,
        transcript_projection::SessionContextCursor {
            branch_id: branch_id.map(str::to_string),
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot;
    RuntimeActiveContext::try_from(&snapshot)
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

fn format_branch_listing(branches: &[transcript_projection::ContextBranchInfo]) -> String {
    branches
        .iter()
        .map(|branch| {
            let marker = if branch.is_current { '*' } else { '-' };
            let mut text = format!("{marker} {}@{}", branch.branch_id, branch.tip_sequence);
            if let Some(label) = &branch.label {
                text.push_str(&format!(" ({label})"));
            }
            text
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

fn branch_dialog_items(branches: &[transcript_projection::ContextBranchInfo]) -> Vec<DialogItem> {
    fn push_children(
        out: &mut Vec<DialogItem>,
        branches: &[transcript_projection::ContextBranchInfo],
        parent: Option<&str>,
        depth: usize,
    ) {
        let mut children = branches
            .iter()
            .filter(|branch| branch.parent_branch_id.as_deref() == parent)
            .collect::<Vec<_>>();
        children.sort_by(|left, right| left.branch_id.cmp(&right.branch_id));

        for branch in children {
            let indent = if depth == 0 {
                String::new()
            } else {
                format!("{}↳ ", "  ".repeat(depth.saturating_sub(1)))
            };
            let mut label = format!("{indent}{}", branch.branch_id);
            if branch.is_current {
                label.push_str(" • current");
            }
            let mut item = DialogItem::new(branch.branch_id.clone(), label, branch.label.clone())
                .with_right_detail(format!("@{}", branch.tip_sequence));
            if depth == 0 {
                item = item.with_section("Context branches");
            }
            out.push(item);
            push_children(out, branches, Some(branch.branch_id.as_str()), depth + 1);
        }
    }

    let mut items = Vec::new();
    push_children(
        &mut items,
        branches,
        Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID),
        1,
    );

    // Include root/main first with no indent.
    if let Some(root) = branches
        .iter()
        .find(|branch| branch.branch_id == crate::transcript::ROOT_CONTEXT_BRANCH_ID)
    {
        let root_item = DialogItem::new(
            root.branch_id.clone(),
            if root.is_current {
                format!("{} • current", root.branch_id)
            } else {
                root.branch_id.clone()
            },
            root.label.clone(),
        )
        .with_section("Context branches")
        .with_right_detail(format!("@{}", root.tip_sequence));
        items.insert(0, root_item);
    }

    items
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

    for metadata in context.view.provider_folded_outputs() {
        items.push(
            DialogItem::new(
                format!("folded:{}", metadata.output_id),
                format!("Folded output {}", metadata.output_id),
                metadata
                    .tool_name
                    .clone()
                    .or_else(|| metadata.stream.clone()),
            )
            .with_section("Folded output"),
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
        "folded" => Some(ContextDetailTarget::FoldedOutput(value.to_string())),
        _ => None,
    }
}

fn context_dialog_target_id(target: &ContextDetailTarget) -> String {
    match target {
        ContextDetailTarget::Node(node_id) => format!("node:{node_id}"),
        ContextDetailTarget::Block(block_id) => format!("block:{block_id}"),
        ContextDetailTarget::Summary(artifact_id) => format!("summary:{artifact_id}"),
        ContextDetailTarget::FoldedOutput(output_id) => format!("folded:{output_id}"),
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
        ContextDetailTarget::FoldedOutput(output_id) => {
            if !folded_output_visible(context, output_id) {
                return None;
            }
            let metadata = context.view.folded_outputs.get(output_id)?;
            let opened = context.view.open_folded_output(
                output_id,
                crate::context_view::DEFAULT_OPEN_CONTENT_MAX_BYTES,
            )?;
            let mut lines = vec![DialogItem::new(
                "size",
                "Open detail",
                Some(format!("{} bytes", opened.returned_bytes)),
            )];
            for (label, value) in folded_output_detail_lines(metadata) {
                lines.push(DialogItem::new(label, label, Some(value)));
            }
            for line in opened.content.lines().take(4) {
                lines.push(DialogItem::new("content", line, None));
            }
            (format!("Folded output {}", metadata.output_id), lines)
        }
    };

    Some(DialogState::new(
        DialogKind::ContextDetail,
        format!("Detail · {title}"),
        None,
        lines,
    ))
}

fn folded_output_visible(context: &super::state::ContextPaneState, output_id: &str) -> bool {
    context.view.folded_outputs.contains_key(output_id)
        && !context.view.is_compacted_folded_output(output_id)
        && !context.view.blocks.values().any(|block| {
            block.folded_output_id.as_deref() == Some(output_id)
                && context.view.view_state.status(&block.block_id)
                    == Some(crate::context_view::ContextViewStatus::RemovedFromView)
        })
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
    if block.folded_output_id.is_some() {
        labels.push("Folded output".into());
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
        crate::context_view::ContextBlockSource::FoldedOutput { .. } => "Folded output",
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
        crate::context_view::ContextBlockSource::FoldedOutput { output_id } => {
            lines.push(("Source", format!("Folded output {output_id}")));
            if let Some(metadata) = view.folded_outputs.get(output_id) {
                lines.extend(folded_output_detail_lines(metadata));
            }
        }
    }
    lines
}

fn folded_output_detail_lines(
    metadata: &crate::context_view::FoldedOutputMetadata,
) -> Vec<(&'static str, String)> {
    let mut lines = Vec::new();
    if let Some(tool_name) = metadata.tool_name.clone() {
        lines.push(("Tool", tool_name));
    }
    if let Some(stream) = metadata.stream.clone() {
        lines.push(("Stream", stream));
    }
    if let Some(command) = metadata.shell_command.clone() {
        lines.push(("Command", truncate_dialog_text(&command)));
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

fn slugify_branch_label(label: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in label.chars().flat_map(|ch| ch.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn next_branch_id(
    branches: &[transcript_projection::ContextBranchInfo],
    label: Option<&str>,
) -> String {
    let existing = branches
        .iter()
        .map(|branch| branch.branch_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let base = label
        .map(slugify_branch_label)
        .filter(|slug| !slug.is_empty())
        .unwrap_or_else(|| "branch".into());
    if !existing.contains(base.as_str()) {
        return base;
    }
    let mut suffix = 2u64;
    loop {
        let candidate = format!("{base}-{suffix}");
        if !existing.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

struct PreparedContextScope {
    state: Arc<StdMutex<crate::transcript::ContextScopeState>>,
    restore_point: Option<(
        crate::transcript::ActiveContextExperiment,
        Vec<crate::protocol_frames::ProtocolFrame>,
        crate::runtime_context::RuntimeSnapshot,
    )>,
}

fn prepare_context_scope(recorder: &TranscriptRecorder) -> Result<PreparedContextScope> {
    let state = recorder.context_scope_state();
    let restore_point = if let Some(scope) = recorder.active_context_experiment() {
        let snapshot = project_runtime_restore_snapshot_with_children(
            recorder.session_id(),
            read_records(recorder.path())?,
            None,
            transcript_projection::SessionContextCursor {
                branch_id: Some(scope.parent_branch_id.clone()),
                leaf_sequence: Some(scope.base_sequence),
            },
            recorder
                .path()
                .parent()
                .ok_or_else(|| anyhow!("transcript path has no parent directory"))?,
        )?;
        RuntimeActiveContext::try_from(&snapshot.snapshot)?;
        Some((scope, snapshot.protocol_frames, snapshot.snapshot))
    } else {
        None
    };
    Ok(PreparedContextScope {
        state,
        restore_point,
    })
}

fn apply_prepared_context_scope<C>(agent: &mut Agent<C>, prepared: PreparedContextScope)
where
    C: Config,
{
    agent.set_context_scope_state(prepared.state);
    if let Some((scope, frames, snapshot)) = prepared.restore_point {
        agent.set_context_experiment_restore_point(scope, frames, snapshot);
    } else {
        agent.clear_context_experiment_restore_point();
    }
}

fn create_context_branch<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    label: Option<String>,
) -> Result<transcript_projection::RuntimeRestoreSnapshot>
where
    C: Config,
{
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    if recorder.active_context_experiment().is_some() {
        return Err(anyhow!(
            "manual branch creation is unavailable during an active context experiment; use context__return first"
        ));
    }
    let records = read_records(recorder.path())?;
    let snapshot = transcript_projection::build_session_context_snapshot(
        recorder.session_id().to_string(),
        records.clone(),
        None,
        transcript_projection::SessionContextCursor {
            branch_id: recorder.current_context_branch_id().map(str::to_string),
            leaf_sequence: None,
        },
    )?;
    let branches = transcript_projection::list_context_branches(
        &records,
        recorder.current_context_branch_id(),
    )?;
    let branch_id = next_branch_id(&branches, label.as_deref());
    let parent_branch_id = snapshot.branch_id.clone();
    let created_sequence = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
    let checkout_sequence = created_sequence
        .checked_add(1)
        .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
    let mut candidate_records = records;
    candidate_records.push(crate::transcript::TranscriptRecord {
        session_id: recorder.session_id().to_string(),
        sequence: created_sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event: crate::transcript::TranscriptEvent::ContextBranchCreated {
            branch_id: branch_id.clone(),
            parent_branch_id: parent_branch_id.clone(),
            base_sequence: snapshot.leaf_sequence,
            label: label.clone(),
        },
    });
    candidate_records.push(crate::transcript::TranscriptRecord {
        session_id: recorder.session_id().to_string(),
        sequence: checkout_sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event: crate::transcript::TranscriptEvent::ContextCheckout {
            branch_id: branch_id.clone(),
            leaf_sequence: snapshot.leaf_sequence,
        },
    });
    let sessions_dir = recorder
        .path()
        .parent()
        .ok_or_else(|| anyhow!("transcript path has no parent directory"))?;
    let post_creation = project_runtime_restore_snapshot_with_children(
        recorder.session_id(),
        candidate_records,
        None,
        transcript_projection::SessionContextCursor {
            branch_id: Some(branch_id.clone()),
            leaf_sequence: None,
        },
        sessions_dir,
    )?;
    RuntimeActiveContext::try_from(&post_creation.snapshot)?;
    if post_creation.snapshot.context_scope_revision != checkout_sequence {
        return Err(anyhow!(
            "branch projection revision does not match checkout sequence"
        ));
    }
    let prepared_scope = prepare_context_scope(&recorder)?;
    recorder.record_context_branch_created(
        branch_id.clone(),
        parent_branch_id,
        snapshot.leaf_sequence,
        label.clone(),
    )?;
    recorder.record_context_checkout(branch_id.clone(), snapshot.leaf_sequence)?;
    sync_recorder_branch(&mut recorder, &branch_id);
    agent
        .restore_runtime_snapshot(
            post_creation.protocol_frames.clone(),
            post_creation.snapshot.clone(),
        )
        .map_err(|error| anyhow!("validated branch snapshot could not be installed: {error}"))?;
    if let Some(model) = &post_creation.latest_model {
        agent.set_model(model.clone());
    }
    agent.restore_turn_sequence(post_creation.max_turn_id);
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(post_creation)
}

fn checkout_context_branch<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    branch_id: &str,
) -> Result<transcript_projection::RuntimeRestoreSnapshot>
where
    C: Config,
{
    let (session_id, records, token_usage, sessions_dir, mut recorder) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        if recorder.active_context_experiment().is_some() {
            return Err(anyhow!(
                "manual branch checkout is unavailable during an active context experiment; use context__return first"
            ));
        }
        let records = read_records(recorder.path())?;
        let token_usage = agent.session_token_usage().ok().map(|usage| {
            TokenUsageEvent::with_breakdown(
                usage.used_tokens,
                usage.context_window_tokens,
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_tokens,
            )
        });
        let sessions_dir = recorder
            .path()
            .parent()
            .map(ToOwned::to_owned)
            .ok_or_else(|| anyhow!("transcript path has no parent directory"))?;
        (
            recorder.session_id().to_string(),
            records,
            token_usage,
            sessions_dir,
            recorder,
        )
    };
    let branch_tip = transcript_projection::list_context_branches(
        &records,
        recorder.current_context_branch_id(),
    )?
    .into_iter()
    .find(|branch| branch.branch_id == branch_id)
    .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?
    .tip_sequence;
    let checkout_sequence = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
    let mut candidate_records = records;
    candidate_records.push(crate::transcript::TranscriptRecord {
        session_id: session_id.clone(),
        sequence: checkout_sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event: crate::transcript::TranscriptEvent::ContextCheckout {
            branch_id: branch_id.to_string(),
            leaf_sequence: branch_tip,
        },
    });
    let snapshot = project_runtime_restore_snapshot_with_children(
        &session_id,
        candidate_records,
        token_usage,
        transcript_projection::SessionContextCursor {
            branch_id: Some(branch_id.to_string()),
            leaf_sequence: None,
        },
        &sessions_dir,
    )?;
    RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    if snapshot.snapshot.context_scope_revision != checkout_sequence {
        return Err(anyhow!(
            "checkout projection revision does not match checkout sequence"
        ));
    }
    recorder.record_context_checkout(branch_id.to_string(), branch_tip)?;
    let persisted = read_records(recorder.path())?;
    let persisted_checkout = persisted
        .last()
        .ok_or_else(|| anyhow!("checkout record missing after persistence"))?;
    if persisted_checkout.session_id != session_id
        || persisted_checkout.sequence != checkout_sequence
        || persisted_checkout.context_branch_id.is_some()
        || !matches!(&persisted_checkout.event, crate::transcript::TranscriptEvent::ContextCheckout { branch_id: persisted_branch, leaf_sequence } if persisted_branch == branch_id && *leaf_sequence == branch_tip)
    {
        return Err(anyhow!(
            "persisted checkout does not match validated candidate"
        ));
    }
    sync_recorder_branch(&mut recorder, &snapshot.branch_id);
    agent.restore_runtime_snapshot(snapshot.protocol_frames.clone(), snapshot.snapshot.clone())?;
    if let Some(model) = &snapshot.latest_model {
        agent.set_model(model.clone());
    }
    agent.restore_turn_sequence(snapshot.max_turn_id);
    Ok(snapshot)
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

fn project_runtime_restore_snapshot_with_children(
    session_id: &str,
    records: Vec<crate::transcript::TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
    cursor: transcript_projection::SessionContextCursor,
    sessions_dir: &std::path::Path,
) -> Result<transcript_projection::RuntimeRestoreSnapshot> {
    let resolved = transcript_projection::project_runtime_restore_snapshot(
        session_id.to_string(),
        records.clone(),
        token_usage.clone(),
        cursor.clone(),
        &[],
    )?;
    let children = list_child_sessions_for_parent(sessions_dir, &resolved.records);
    transcript_projection::project_runtime_restore_snapshot(
        session_id.to_string(),
        records,
        token_usage,
        cursor,
        &children,
    )
}

fn send_parent_session_view(
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<()> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder.current_context_branch_id().map(str::to_string),
        )
    };
    let snapshot = project_runtime_restore_snapshot_with_children(
        &session_id,
        records,
        None,
        transcript_projection::SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
        &sessions_dir_for_transcript(transcript)?,
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    let evidence_count = snapshot.snapshot.evidence.len();
    let _ = runner_tx.send(RunnerEvent::SessionResumed {
        session_id: snapshot.session_id,
        branch_id: snapshot.branch_id,
        messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
        records: snapshot.records,
        evidence_count,
        model_id: snapshot.latest_model,
        token_usage: snapshot.token_usage,
        runtime_context,
    });
    Ok(())
}

fn restored_messages_from_protocol_frames(
    protocol_frames: &[crate::protocol_frames::ProtocolFrame],
) -> Vec<crate::agent::ConversationMessage> {
    crate::protocol_frames::history_items_from_frames(protocol_frames)
        .into_iter()
        .filter_map(|item| match item {
            crate::request_builder::HistoryItem::ContextSummary { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Summary,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::UserMessage { content } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: content.display_text(),
                })
            }
            crate::request_builder::HistoryItem::InternalContinuation { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::User,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantText { text } => {
                Some(crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content: text,
                })
            }
            crate::request_builder::HistoryItem::AssistantToolCalls { text, .. } => {
                text.map(|content| crate::agent::ConversationMessage {
                    role: crate::agent::ConversationRole::Assistant,
                    content,
                })
            }
            crate::request_builder::HistoryItem::ToolOutput { .. } => None,
        })
        .collect()
}

fn send_child_session_view(
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    sessions_dir: &std::path::Path,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    active_child: Option<crate::transcript::ChildSessionSummary>,
    navigation: ChildNavigation,
    anchor_child_session_id: Option<&str>,
) -> Result<Option<String>> {
    let (parent_session_id, parent_records) = current_session_records(transcript)?;
    let mut children = list_child_sessions_for_parent(sessions_dir, &parent_records);
    if let Some(ref active_child) = active_child
        && active_child.parent_session_id == parent_session_id
        && !children
            .iter()
            .any(|child| child.child_session_id == active_child.child_session_id)
    {
        children.push(active_child.clone());
        sort_child_session_summaries(&mut children);
    }
    if children.is_empty() {
        let _ = runner_tx.send(RunnerEvent::Status(
            "No child subagent transcripts for this session".into(),
        ));
        return Ok(None);
    }

    let current_index = anchor_child_session_id.and_then(|child_session_id| {
        children
            .iter()
            .position(|child| child.child_session_id == child_session_id)
    });

    let index = match navigation {
        ChildNavigation::First => 0,
        ChildNavigation::Next => current_index
            .map(|index| (index + 1) % children.len())
            .unwrap_or(0),
        ChildNavigation::Prev => current_index
            .map(|index| {
                if index == 0 {
                    children.len() - 1
                } else {
                    index - 1
                }
            })
            .unwrap_or(children.len() - 1),
    };
    let child = &children[index];
    let records =
        read_child_session_records_allow_partial_tail(sessions_dir, &child.child_session_id)?;
    let snapshot = project_runtime_restore_snapshot_with_children(
        &child.child_session_id,
        records,
        None,
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        sessions_dir,
    )?;
    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
    let _ = runner_tx.send(RunnerEvent::ChildSessionViewed {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        records: snapshot.records,
        runtime_context,
    });
    Ok(Some(child.child_session_id.clone()))
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
    let children = list_child_sessions_for_parent(sessions_dir, &parent_records);
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

    for (call_id, name) in &interrupt.parent_tool_calls {
        let _ = recorder.record_tool_call_cancelled(call_id.clone(), name.clone());
    }

    let turn_id = read_records(recorder.path())
        .ok()
        .map(|records| restore_max_turn_id(&records))
        .filter(|turn_id| *turn_id > 0);
    let _ = recorder.record_turn_interrupted(turn_id);
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
        None,
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
    let prepared_scope = prepare_context_scope(&recorder)?;
    agent.restore_runtime_snapshot(snapshot.protocol_frames, snapshot.snapshot)?;
    agent.restore_turn_sequence(max_turn_id);
    sync_recorder_branch(&mut recorder, &branch_id);
    apply_prepared_context_scope(agent, prepared_scope);
    Ok(())
}

fn sync_agent_context_scope_from_recorder<C>(
    agent: &mut Agent<C>,
    recorder: &TranscriptRecorder,
) -> Result<()>
where
    C: Config,
{
    agent.set_context_scope_state(recorder.context_scope_state());
    if let Some(scope) = recorder.active_context_experiment() {
        let snapshot = project_runtime_restore_snapshot_with_children(
            recorder.session_id(),
            read_records(recorder.path())?,
            None,
            transcript_projection::SessionContextCursor {
                branch_id: Some(scope.parent_branch_id.clone()),
                leaf_sequence: Some(scope.base_sequence),
            },
            recorder
                .path()
                .parent()
                .ok_or_else(|| anyhow::anyhow!("transcript path has no parent directory"))?,
        )?;
        agent.set_context_experiment_restore_point(
            scope,
            snapshot.protocol_frames,
            snapshot.snapshot,
        );
    } else {
        agent.clear_context_experiment_restore_point();
    }
    Ok(())
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
    mcp_tools_rx: Option<mpsc::UnboundedReceiver<anyhow::Result<Vec<mcp::McpTool>>>>,
) -> Result<()>
where
    C: Config + Clone + Send + Sync + 'static,
{
    {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        sync_agent_context_scope_from_recorder(&mut agent, &recorder)?;
    }
    let model_id = agent.model().to_string();
    let model_label = agent.model().to_string();
    let permission_mode_label = agent.permission_mode().to_string();
    let mut state = TuiState::new(model_id, model_label, permission_mode_label);
    let preferences = TuiPreferences::load_from_dir(&preferences_dir);
    state.set_tool_output_expanded(preferences.tool_output_expanded);
    state.set_transcript_scrollbar_visible(preferences.transcript_scrollbar_visible);
    state.set_provider_label(provider_label);

    if let Some(active_model) = available_models
        .iter()
        .find(|model| model.id == state.model_id)
    {
        state.set_model(active_model.id.clone(), active_model.label.clone());
        state.set_model_context_window(active_model.context_window_tokens);
        state.set_reasoning_effort_label(Some(reasoning_effort_status_label(
            active_model.reasoning_effort,
        )));
    }

    if !api_key_configured {
        state.set_footer("Missing API key", Some(api_key_hint.clone()));
    }

    if let Some(toast) = startup_toast {
        state.show_toast(toast.message, toast.kind);
    }

    let (runner_tx, runner_rx) = mpsc::unbounded_channel();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<RunnerCommand>();
    let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<InterruptRequest>();
    let subagent_runtime = SubagentRuntime::new();
    let cleanup_subagent_runtime = subagent_runtime.clone();
    let mut runtime = TuiRuntime::new(
        state,
        runner_rx,
        available_models,
        sessions_dir.clone(),
        preferences_dir,
    );
    let mut terminal = OwnedTerminal::new()?;
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
        let subagent_runtime = subagent_runtime;
        let mut deferred_command: Option<RunnerCommand> = None;

        loop {
            tokio::select! {
                command = async {
                    if deferred_command.is_some() {
                        deferred_command.take()
                    } else {
                        prompt_rx.recv().await
                    }
                } => {
                    let Some(command) = command else {
                        break;
                    };

                    let prompt = match command {
                        RunnerCommand::Prompt(prompt) => prompt,
                        RunnerCommand::ShowBranchTree => {
                            let branches = match transcript.lock() {
                                Ok(recorder) => match read_records(recorder.path()).and_then(|records| {
                                    transcript_projection::list_context_branches(
                                        &records,
                                        recorder.current_context_branch_id(),
                                    )
                                    .map_err(Into::into)
                                }) {
                                    Ok(branches) => branches,
                                    Err(error) => {
                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                            "failed to load context tree: {error}"
                                        ))));
                                        continue;
                                    }
                                },
                                Err(_) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        "transcript recorder poisoned",
                                    )));
                                    continue;
                                }
                            };
                            let _ = runner_tx.send(RunnerEvent::ContextBranchesLoaded { branches });
                            continue;
                        }
                        RunnerCommand::ListBranches => {
                            let message = match transcript.lock() {
                                Ok(recorder) => match read_records(recorder.path()).and_then(|records| {
                                    transcript_projection::list_context_branches(
                                        &records,
                                        recorder.current_context_branch_id(),
                                    )
                                    .map_err(Into::into)
                                }) {
                                    Ok(branches) => format_branch_listing(&branches),
                                    Err(error) => {
                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                            "failed to list context branches: {error}"
                                        ))));
                                        continue;
                                    }
                                },
                                Err(_) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        "transcript recorder poisoned",
                                    )));
                                    continue;
                                }
                            };
                            let _ = runner_tx.send(RunnerEvent::Status(message));
                            continue;
                        }
                        RunnerCommand::CreateBranch { label } => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Status(
                                    "Wait for the active subagent to finish before creating a context branch".into(),
                                ));
                                continue;
                            }
                            match create_context_branch(&mut agent, &transcript, label.clone()) {
                                Ok(snapshot) => {
                                    let branch_id = snapshot.branch_id.clone();
                                    let leaf_sequence = snapshot.leaf_sequence;
                                    let _ = runner_tx.send(RunnerEvent::ContextBranchChanged {
                                        branch_id: branch_id.clone(),
                                    });
                                    let context = match RuntimeActiveContext::try_from(&snapshot.snapshot) {
                                        Ok(context) => context,
                                        Err(error) => {
                                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                "validated branch context could not be emitted: {error}"
                                            ))));
                                            continue;
                                        }
                                    };
                                    let _ = runner_tx.send(RunnerEvent::RuntimeContextUpdated(
                                        RuntimeContextUpdatedEvent {
                                            context,
                                            disposition: RuntimeContextDisposition::ReplaceScope,
                                        },
                                    ));
                                    let detail = label
                                        .filter(|value| !value.trim().is_empty())
                                        .map(|value| format!("{branch_id} @ {leaf_sequence} · {value}"))
                                        .unwrap_or_else(|| format!("{branch_id} @ {leaf_sequence}"));
                                    let _ = runner_tx.send(RunnerEvent::Status(format!(
                                        "Created and checked out {detail}"
                                    )));
                                }
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to create context branch: {error}"
                                    ))));
                                }
                            }
                            continue;
                        }
                        RunnerCommand::CheckoutBranch(branch_id) => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Status(
                                    "Wait for the active subagent to finish before checking out another context branch".into(),
                                ));
                                continue;
                            }
                            match checkout_context_branch(&mut agent, &transcript, &branch_id) {
                                Ok(snapshot) => {
                                    let evidence_count = snapshot.snapshot.evidence.len();
                                    let branch_label = snapshot.branch_id.clone();
                                    let leaf_sequence = snapshot.leaf_sequence;
                                    let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)
                                        .expect("validated checkout snapshot must have active context");
                                    let _ = runner_tx.send(RunnerEvent::SessionResumed {
                                        session_id: snapshot.session_id,
                                        branch_id: snapshot.branch_id,
                                        messages: restored_messages_from_protocol_frames(
                                            &snapshot.protocol_frames,
                                        ),
                                        records: snapshot.records,
                                        evidence_count,
                                        model_id: snapshot.latest_model,
                                        token_usage: snapshot.token_usage,
                                        runtime_context,
                                    });
                                    let _ = runner_tx.send(RunnerEvent::Status(format!(
                                        "Checked out {branch_label} @ {leaf_sequence}"
                                    )));
                                }
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to checkout context branch: {error}"
                                    ))));
                                }
                            }
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
                                },
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        format!("{error:#}"),
                                    )));
                                    let _ = runner_tx.send(RunnerEvent::Done);
                                    continue;
                                }
                            };

                            let interrupted_child_session_id = {
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
                                    Some(crate::tui::runner::subagent_event_sender::<C>(
                                        runner_tx.clone(),
                                    )),
                                );

                                tokio::pin!(delegate);
                                let mut interrupted = false;
                                let mut interrupted_child_session_id = None;

                                loop {
                                    tokio::select! {
                                        biased;
                                        result = &mut delegate => {
                                            match result {
                                                Ok(_) => {
                                                    let _ = runner_tx.send(RunnerEvent::Done);
                                                }
                                                Err(error) => {
                                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!("{error:#}"))));
                                                    let _ = runner_tx.send(RunnerEvent::Done);
                                                }
                                            }
                                            break;
                                        }
                                        Some(interrupt) = cancel_rx.recv() => {
                                            interrupted = true;
                                            interrupted_child_session_id = subagent_runtime
                                                .active_child()
                                                .map(|child| child.child_session_id);
                                            subagent_runtime.cancel_active();
                                            record_interrupt_transcript(&transcript, &interrupt);
                                            if let Err(error) = delegate.await {
                                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!("{error:#}"))));
                                            }
                                            break;
                                        }
                                        command = prompt_rx.recv() => {
                                            match command {
                                                Some(RunnerCommand::Prompt(prompt)) => {
                                                    deferred_command = Some(RunnerCommand::Prompt(prompt));
                                                    let _ = runner_tx.send(RunnerEvent::AssistantDone { message_id: None });
                                                    break;
                                                }
                                                Some(RunnerCommand::ViewChild { navigation, anchor_child_session_id }) => {
                                                    match send_child_session_view(
                                                        &runner_tx,
                                                        &sessions_dir,
                                                        &transcript,
                                                        subagent_runtime.active_child(),
                                                        navigation,
                                                        anchor_child_session_id.as_deref(),
                                                    ) {
                                                        Ok(_) => {}
                                                        Err(error) => {
                                                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                                "failed to view child transcript: {error}"
                                                            ))));
                                                        }
                                                    }
                                                }
                                                Some(RunnerCommand::ViewParent) => {
                                                    if let Err(error) = send_parent_session_view(&runner_tx, &transcript) {
                                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                            "failed to view parent transcript: {error}"
                                                        ))));
                                                    }
                                                }
                                                Some(_) => {}
                                                None => break,
                                            }
                                        }
                                    }
                                }

                                interrupted.then_some(interrupted_child_session_id)
                            };

                            if let Some(interrupted_child_session_id) = interrupted_child_session_id {
                                if let Err(error) = rehydrate_agent_from_transcript(&mut agent, &transcript) {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore interrupted session context: {error}"
                                    ))));
                                }
                                send_subagent_interrupted(&runner_tx, interrupted_child_session_id);
                            }
                            continue;
                        }
                        RunnerCommand::ViewChild { navigation, anchor_child_session_id } => {
                            match send_child_session_view(
                                &runner_tx,
                                &sessions_dir,
                                &transcript,
                                subagent_runtime.active_child(),
                                navigation,
                                anchor_child_session_id.as_deref(),
                            ) {
                                Ok(_) => {}
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to view child transcript: {error}"
                                    ))));
                                }
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
                                let _ = runner_tx.send(RunnerEvent::Status(
                                    "Wait for the active subagent to finish before compacting context".into(),
                                ));
                                let _ = runner_tx.send(RunnerEvent::Done);
                                continue;
                            }

                            let transcript = transcript.clone();
                            let compaction_started = Arc::new(AtomicBool::new(false));
                            let compaction_streamed = Arc::new(AtomicBool::new(false));
                            let compacted_summary = Arc::new(StdMutex::new(None::<String>));
                            let start_runner_tx = runner_tx.clone();
                            let start_flag = Arc::clone(&compaction_started);
                            let mut on_start = move || {
                                if !start_flag.swap(true, Ordering::AcqRel) {
                                    let _ = start_runner_tx.send(RunnerEvent::Notice(
                                        NoticeEvent::new(compaction_separator(
                                            COMPACTION_SEPARATOR_LABEL,
                                        )),
                                    ));
                                }
                                Ok(())
                            };
                            let delta_runner_tx = runner_tx.clone();
                            let delta_streamed = Arc::clone(&compaction_streamed);
                            let mut on_delta = move |delta: &str| {
                                let delta = delta.to_string();
                                delta_streamed.store(true, Ordering::Release);
                                let _ = delta_runner_tx.send(RunnerEvent::AssistantDelta(
                                    AssistantDeltaEvent::with_message_id(
                                        COMPACTION_MESSAGE_ID,
                                        delta,
                                    ),
                                ));
                                Ok(())
                            };
                            let event_summary = Arc::clone(&compacted_summary);
                            let on_event = |event| {
                                let transcript = transcript.clone();
                                let event_summary = Arc::clone(&event_summary);
                                async move {
                                    if let AgentEvent::ContextCompacted(event) = event {
                                        if let Ok(mut summary) = event_summary.lock() {
                                            *summary = Some(event.summary.clone());
                                        }
                                        let mut recorder = transcript.lock().map_err(|_| {
                                            anyhow::anyhow!("transcript recorder poisoned")
                                        })?;
                                        recorder.record_context_compaction(event)?;
                                    }
                                    Ok(())
                                }
                            };
                            match agent
                                .compact_session_stream_async(
                                    on_event,
                                    &mut on_start,
                                    &mut on_delta,
                                )
                                .await
                            {
                                Ok(ManualCompactionOutcome::Compacted { retained_items }) => {
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
                                                "failed to refresh compacted context: {error}"
                                            ))));
                                            let _ = runner_tx.send(RunnerEvent::Done);
                                            continue;
                                        }
                                    }
                                    if !compaction_started.swap(true, Ordering::AcqRel) {
                                        let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::new(
                                            compaction_separator(COMPACTION_SEPARATOR_LABEL),
                                        )));
                                    }
                                    if !compaction_streamed.load(Ordering::Acquire) {
                                        if let Ok(summary) = compacted_summary.lock()
                                            && let Some(summary) = summary.as_ref()
                                        {
                                            let _ = runner_tx.send(RunnerEvent::AssistantDelta(
                                                AssistantDeltaEvent::with_message_id(
                                                    COMPACTION_MESSAGE_ID,
                                                    summary.clone(),
                                                ),
                                            ));
                                        }
                                    }
                                    let _ = runner_tx.send(RunnerEvent::AssistantDone {
                                        message_id: Some(COMPACTION_MESSAGE_ID.into()),
                                    });
                                    let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::new(
                                        compaction_separator(COMPACTION_SEPARATOR_LABEL),
                                    )));
                                    let _ = runner_tx.send(RunnerEvent::Status(format!(
                                        "Context compacted ({retained_items} history items retained)"
                                    )));
                                }
                                Ok(ManualCompactionOutcome::NothingToCompact) => {
                                    let _ = runner_tx.send(RunnerEvent::Notice(NoticeEvent::new(
                                        "Nothing to compact yet",
                                    )));
                                    let _ = runner_tx
                                        .send(RunnerEvent::Status("Nothing to compact yet".into()));
                                }
                                Err(error) => {
                                    if compaction_started.load(Ordering::Acquire) {
                                        let _ = runner_tx.send(RunnerEvent::AssistantDone {
                                            message_id: Some(COMPACTION_MESSAGE_ID.into()),
                                        });
                                    }
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to compact context: {error}"
                                    ))));
                                }
                            }
                            let _ = runner_tx.send(RunnerEvent::Done);
                            continue;
                        }
                        RunnerCommand::ViewParent => {
                            if let Err(error) = send_parent_session_view(&runner_tx, &transcript) {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to view parent transcript: {error}"
                                ))));
                            }
                            continue;
                        }
                        RunnerCommand::SetPermissionMode(mode) => {
                            let previous = agent.permission_mode();
                            if previous != mode {
                                let previous_label = previous.to_string();
                                let new_label = mode.to_string();
                                agent.set_permission_mode(mode);
                                let _ = runner.record_permission_mode_changed(&previous_label, &new_label);
                            } else {
                                agent.set_permission_mode(mode);
                            }
                            continue;
                        }
                        RunnerCommand::SetModel(model) => {
                            let previous = agent.model().to_string();
                            if previous != model {
                                agent.set_model(model.clone());
                                let _ = runner.record_model_changed(&previous, &model);
                            } else {
                                agent.set_model(model);
                            }
                            continue;
                        }
                        RunnerCommand::SetReasoningEffort(effort) => {
                            agent.set_reasoning_effort(effort);
                            continue;
                        }
                        RunnerCommand::ResumeSession(prefix) => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Status(
                                    "Wait for the active subagent to finish before resuming another session".into(),
                                ));
                                continue;
                            }

                            let sessions = match crate::transcript::list_sessions(&sessions_dir) {
                                Ok(sessions) => sessions,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to list sessions: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let session_id = match crate::transcript::resolve_session_id(&sessions, &prefix)
                            {
                                Ok(session_id) => session_id,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to resolve session: {error:?}"
                                    ))));
                                    continue;
                                }
                            };
                            let records = match crate::transcript::read_records(
                                sessions_dir.join(format!("{session_id}.jsonl")),
                            ) {
                                Ok(records) => records,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to read session: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let token_usage = match agent.session_token_usage() {
                                Ok(usage) => Some(TokenUsageEvent::with_breakdown(
                                    usage.used_tokens,
                                    usage.context_window_tokens,
                                    usage.input_tokens,
                                    usage.output_tokens,
                                    usage.cached_tokens,
                                )),
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore session token usage: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let snapshot = match project_runtime_restore_snapshot_with_children(
                                &session_id,
                                records,
                                token_usage,
                                transcript_projection::SessionContextCursor {
                                    branch_id: None,
                                    leaf_sequence: None,
                                },
                                &sessions_dir,
                            ) {
                                Ok(snapshot) => snapshot,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore session evidence: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let runtime_context = match RuntimeActiveContext::try_from(&snapshot.snapshot) {
                                Ok(context) => context,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to validate restored session context: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let mut new_recorder = match TranscriptRecorder::open_existing(&sessions_dir, &session_id) {
                                Ok(recorder) => recorder,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                        format!("failed to open session transcript: {error}"),
                                    )));
                                    continue;
                                }
                            };
                            sync_recorder_branch(&mut new_recorder, &snapshot.branch_id);
                            let prepared_scope = match prepare_context_scope(&new_recorder) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to prepare restored context scope: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let mut recorder = match transcript.lock() {
                                Ok(recorder) => recorder,
                                Err(_) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new("transcript recorder poisoned")));
                                    continue;
                                }
                            };
                            let old_empty_session_path = empty_session_path(recorder.path());
                            if let Err(error) = agent.restore_runtime_snapshot(
                                snapshot.protocol_frames.clone(),
                                snapshot.snapshot.clone(),
                            ) {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to restore session context: {error}"
                                ))));
                                continue;
                            }
                            if let Some(model) = &snapshot.latest_model {
                                agent.set_model(model.clone());
                            }
                            agent.restore_turn_sequence(snapshot.max_turn_id);
                            apply_prepared_context_scope(&mut agent, prepared_scope);
                            // The shared recorder lock is held across the Agent restore.
                            if {
                                *recorder = new_recorder;
                                true
                            } {
                                ()
                            } else {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                    "transcript recorder poisoned",
                                )));
                                continue;
                            }
                            drop(recorder);
                            if let Some(path) = old_empty_session_path
                                && path != sessions_dir.join(format!("{session_id}.jsonl"))
                            {
                                let _ = std::fs::remove_file(path);
                            }
                            let evidence_count = snapshot.snapshot.evidence.len();
                            let _ = runner_tx.send(RunnerEvent::SessionResumed {
                                session_id: snapshot.session_id,
                                branch_id: snapshot.branch_id,
                                messages: restored_messages_from_protocol_frames(
                                    &snapshot.protocol_frames,
                                ),
                                records: snapshot.records,
                                evidence_count,
                                model_id: snapshot.latest_model,
                                token_usage: snapshot.token_usage,
                                runtime_context,
                            });
                            continue;
                        }
                        RunnerCommand::NewSession => {
                            if subagent_runtime.is_running() {
                                let _ = runner_tx.send(RunnerEvent::Status(
                                    "Wait for the active subagent to finish before starting a new session".into(),
                                ));
                                continue;
                            }

                            let mut new_recorder = match TranscriptRecorder::create(&sessions_dir) {
                                Ok(recorder) => recorder,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to create session transcript: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            if let Err(error) =
                                new_recorder.record_session_started(agent.model().to_string())
                            {
                                let _ = remove_empty_session_file(new_recorder.path());
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to record session start: {error}"
                                ))));
                                continue;
                            }
                            new_recorder.set_current_context_branch_id(None);
                            let session_id = new_recorder.session_id().to_string();
                            let new_path = new_recorder.path().to_path_buf();
                            let records = match read_records(&new_path) {
                                Ok(records) => records,
                                Err(error) => {
                                    let _ = remove_empty_session_file(&new_path);
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to read new session transcript: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let snapshot = match project_runtime_restore_snapshot_with_children(
                                &session_id,
                                records,
                                None,
                                transcript_projection::SessionContextCursor {
                                    branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
                                    leaf_sequence: None,
                                },
                                &sessions_dir,
                            ) {
                                Ok(snapshot) => snapshot,
                                Err(error) => {
                                    let _ = remove_empty_session_file(&new_path);
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to project new session context: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let runtime_context = match RuntimeActiveContext::try_from(&snapshot.snapshot) {
                                Ok(context) => context,
                                Err(error) => {
                                    let _ = remove_empty_session_file(&new_path);
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to validate new session context: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let prepared_scope = match prepare_context_scope(&new_recorder) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = remove_empty_session_file(&new_path);
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!("failed to prepare new session context scope: {error}"))));
                                    continue;
                                }
                            };
                            let mut recorder = match transcript.lock() {
                                Ok(recorder) => recorder,
                                Err(_) => {
                                    let _ = remove_empty_session_file(&new_path);
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new("transcript recorder poisoned")));
                                    continue;
                                }
                            };
                            let old_empty_session_path = empty_session_path(recorder.path());
                            if let Err(error) = agent.restore_new_session_runtime_snapshot(
                                snapshot.protocol_frames.clone(), snapshot.snapshot.clone(), snapshot.max_turn_id,
                            ) {
                                let _ = remove_empty_session_file(&new_path);
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to reset context scope state: {error}"
                                ))));
                                continue;
                            }
                            apply_prepared_context_scope(&mut agent, prepared_scope);
                            if {
                                *recorder = new_recorder;
                                true
                            } {
                                ()
                            } else {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                    "transcript recorder poisoned",
                                )));
                                continue;
                            }
                            drop(recorder);
                            if let Some(path) = old_empty_session_path
                                && path != new_path
                            {
                                let _ = std::fs::remove_file(path);
                            }
                            let _ = runner_tx.send(RunnerEvent::SessionStarted {
                                session_id,
                                records: snapshot.records,
                                runtime_context,
                            });
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
                            tokio::select! {
                                _ = &mut run => break,
                                command = prompt_rx.recv() => {
                                    match command {
                                        Some(RunnerCommand::Prompt(prompt)) => {
                                            deferred_command = Some(RunnerCommand::Prompt(prompt));
                                            let _ = runner_tx.send(RunnerEvent::AssistantDone { message_id: None });
                                            break;
                                        }
                                        Some(RunnerCommand::ViewChild {
                                            navigation,
                                            anchor_child_session_id,
                                        }) => {
                                            match send_child_session_view(
                                                &runner_tx,
                                                &sessions_dir,
                                                &transcript,
                                                subagent_runtime.active_child(),
                                                navigation,
                                                anchor_child_session_id.as_deref(),
                                            ) {
                                                Ok(_) => {}
                                                Err(error) => {
                                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                        "failed to view child transcript: {error}"
                                                    ))));
                                                }
                                            }
                                        }
                                        Some(RunnerCommand::ViewParent) => {
                                            if let Err(error) = send_parent_session_view(&runner_tx, &transcript) {
                                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                    "failed to view parent transcript: {error}"
                                                ))));
                                            }
                                        }
                                        Some(_) => {
                                            let _ = runner_tx.send(RunnerEvent::Status(
                                                "Turn still running · navigation only".into(),
                                            ));
                                        }
                                        None => break,
                                    }
                                }
                                Some(interrupt) = cancel_rx.recv() => {
                                    interrupted = Some(interrupt);
                                    break;
                                }
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

                    match discovery {
                        Ok(tools) => {
                            for tool in tools {
                                let tool_name = tool.name().to_string();
                                if let Err(error) = agent.try_register_tool(tool) {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to register MCP tool '{tool_name}': {error}"
                                    ))));
                                }
                            }
                        }
                        Err(error) => {
                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                "failed to discover MCP tools: {error}"
                            ))));
                        }
                    }
                }
            }
        }
    });

    loop {
        runtime.try_drain_runner_events();
        if let Some(command) = runtime.take_next_queued_prompt_command() {
            command_dispatch::dispatch_command(&mut runtime, command, &prompt_tx, &cancel_tx, true);
        }
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
                            &prompt_tx,
                            &cancel_tx,
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
                            &prompt_tx,
                            &cancel_tx,
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
                            &prompt_tx,
                            &cancel_tx,
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

    drop(prompt_tx);
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
}

impl RuntimeDrawer for TerminalDrawer<'_> {
    fn draw(&mut self, state: &mut TuiState) -> io::Result<()> {
        self.terminal
            .terminal_mut()
            .draw(|frame| render::render(frame, state))
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, CacheUsageReport, TodoItem, TodoStatus};
    use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
    };
    use crate::request_builder::HistoryItem;
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::tui::{
        AppEvent, AppPhase, PermissionDecision, PermissionRequestEvent, PermissionResolutionEvent,
        PermissionResponse, RunnerEvent, RunnerPermissionRequest, TimelineItem, ToolFinishedEvent,
        ToolOutcome, UserMessageEvent,
    };
    use async_openai::{Client, config::OpenAIConfig};
    use std::collections::BTreeMap;
    use tokio::sync::{mpsc, oneshot};

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
                folded_output_id: None,
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
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
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
        assert_eq!(runtime.state().footer_status.summary, "Submitting prompt");
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

        assert_eq!(
            runtime.state().footer_status.summary,
            "Context item unavailable"
        );
        assert_eq!(
            runtime.state().footer_status.detail.as_deref(),
            Some("Refresh context and try again")
        );
        assert!(runtime.state().toast().is_some());
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(crate::tui::TimelineItem::Notice(notice))
                if notice.message == "Context item unavailable · Refresh context and try again"
        ));
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

        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Notice(notice))
                if notice.message == "Context item unavailable · Refresh context and try again"
        ));
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

        assert_eq!(
            runtime.state().footer_status.summary,
            "Context item unavailable"
        );
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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Answer every question before confirming"
        );
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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Custom answer closed"
        );
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
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "child-session".into(),
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
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "other-child".into(),
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
        );

        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: "child-session".into(),
            event: AppEvent::Interrupted,
        });

        assert!(runtime.state().timeline.items().is_empty());
        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Notice(message)) if message.message == "Interrupted by user"
        ));
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
    fn double_escape_confirms_running_turn_interrupt() {
        let mut runtime = runtime();
        runtime.state.phase = AppPhase::Running;

        let first = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("first interrupt hint succeeds");
        assert_eq!(first, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Press Esc again to interrupt"
        );

        let second = runtime
            .handle_input_action(InputAction::Interrupt)
            .expect("second interrupt returns command");
        assert_eq!(second, Some(RuntimeCommand::Interrupt));
        assert_eq!(runtime.state().footer_status.summary, "Interrupting");
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
        assert_eq!(runtime.state().footer_status.summary, "Interrupted");
        assert!(runtime.state().pending_permission.is_none());
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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Press Esc again to interrupt"
        );
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
        assert_eq!(runtime.state().footer_status.summary, "Interrupted");
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
        );

        send_subagent_interrupted(&tx, Some("child-session".into()));
        runtime.try_drain_runner_events();

        assert!(!runtime.runner_turn_active);
        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert!(matches!(
            runtime.state().active_timeline().items().last(),
            Some(crate::tui::TimelineItem::Notice(message)) if message.message == "Interrupted by user"
        ));
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
            );
            runtime.state_mut().set_input(input);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("submit succeeds");

            assert_eq!(command, None, "{input}");
            assert!(runtime.submitted_prompts().is_empty(), "{input}");
            assert_eq!(
                runtime.state().footer_status.summary,
                "Viewing child transcript",
                "{input}"
            );
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
        );

        let blocked = runtime
            .handle_input_action(InputAction::CycleReasoningEffort)
            .expect("shortcut succeeds");
        assert_eq!(blocked, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Viewing child transcript"
        );

        runtime.state_mut().set_input("/child next");
        let child = runtime
            .handle_input_action(InputAction::Submit)
            .expect("child navigation succeeds");
        assert_eq!(
            child,
            Some(RuntimeCommand::ViewChild(ChildNavigation::Next))
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
            Some(RuntimeCommand::ViewChild(ChildNavigation::First))
        );
    }

    #[test]
    fn running_turn_queues_plain_prompts() {
        let mut runtime = runtime();
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
        assert_eq!(runtime.state().footer_status.summary, "Queued prompt");
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
        assert_eq!(runtime.state().footer_status.summary, "Turn still running");
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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Submitting queued prompt"
        );

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
    fn status_runner_event_updates_footer_without_timeline_noise() {
        let mut runtime = runtime();
        runtime.apply_runner_event(RunnerEvent::Status("Explorer started".into()));

        assert_eq!(runtime.state().footer_status.summary, "Explorer started");
        assert!(runtime.state().timeline.items().is_empty());
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
            Some(RuntimeCommand::ViewChild(ChildNavigation::First))
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
        );

        let next = runtime
            .handle_input_action(InputAction::ChildNext)
            .expect("next succeeds");
        assert_eq!(next, Some(RuntimeCommand::ViewChild(ChildNavigation::Next)));

        let prev = runtime
            .handle_input_action(InputAction::ChildPrev)
            .expect("prev succeeds");
        assert_eq!(prev, Some(RuntimeCommand::ViewChild(ChildNavigation::Prev)));

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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Commands: /help, /exit, /quit, /model, /reasoning, /permission, /tool-output, /scrollbar, /compact, /tree, /branches, /branch, /checkout, /resume, /new, /context, /child, /parent · Delegation: @explorer <task>, @fixer <task>, @oracle <task>, @designer <task>, @librarian <task>, @general <task>"
        );
    }

    #[test]
    fn branch_commands_parse_to_runtime_commands() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/branches");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("branches command"),
            Some(RuntimeCommand::ListBranches)
        );

        runtime.state_mut().set_input("/branch feature alpha");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("branch command"),
            Some(RuntimeCommand::CreateBranch {
                label: Some("feature alpha".into())
            })
        );

        runtime.state_mut().set_input("/checkout feature-alpha");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("checkout command"),
            Some(RuntimeCommand::CheckoutBranch("feature-alpha".into()))
        );

        runtime.state_mut().set_input("/tree");
        assert_eq!(
            runtime
                .handle_input_action(InputAction::Submit)
                .expect("tree command"),
            Some(RuntimeCommand::ShowBranchTree)
        );
    }

    #[test]
    fn context_branches_loaded_opens_branch_picker_dialog() {
        let mut runtime = runtime();
        runtime.apply_runner_event(RunnerEvent::ContextBranchesLoaded {
            branches: vec![
                transcript_projection::ContextBranchInfo {
                    branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                    parent_branch_id: None,
                    label: None,
                    tip_sequence: 2,
                    is_current: true,
                },
                transcript_projection::ContextBranchInfo {
                    branch_id: "feature-a".into(),
                    parent_branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
                    label: Some("Feature A".into()),
                    tip_sequence: 5,
                    is_current: false,
                },
            ],
        });

        let dialog = runtime.state().dialog().expect("branch picker should open");
        assert_eq!(dialog.kind, DialogKind::BranchPicker);
        assert_eq!(dialog.title, "Context tree");
        assert_eq!(
            dialog.items[0].id,
            crate::transcript::ROOT_CONTEXT_BRANCH_ID
        );
        assert!(dialog.items.iter().any(|item| item.id == "feature-a"));
    }

    #[test]
    fn create_branch_writes_metadata_and_switches_branch_context() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-branch-create-test-{}",
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
        recorder.record_user_message("hello").expect("user message");
        let path = recorder.path().to_path_buf();
        let transcript = Arc::new(StdMutex::new(recorder));

        let mut agent = test_agent();
        let snapshot = create_context_branch(&mut agent, &transcript, Some("Feature Alpha".into()))
            .expect("create branch");
        let branch_id = snapshot.branch_id;
        let leaf_sequence = snapshot.leaf_sequence;

        assert_eq!(branch_id, "feature-alpha");
        assert_eq!(leaf_sequence, 2);
        assert_eq!(
            transcript
                .lock()
                .expect("recorder lock")
                .current_context_branch_id(),
            Some("feature-alpha")
        );

        let records = read_records(&path).expect("read transcript");
        assert!(matches!(
            &records[2].event,
            TranscriptEvent::ContextBranchCreated { branch_id, parent_branch_id, base_sequence, label }
                if branch_id == "feature-alpha"
                    && parent_branch_id == crate::transcript::ROOT_CONTEXT_BRANCH_ID
                    && *base_sequence == 2
                    && label.as_deref() == Some("Feature Alpha")
        ));
        assert!(matches!(
            &records[3].event,
            TranscriptEvent::ContextCheckout { branch_id, leaf_sequence }
                if branch_id == "feature-alpha" && *leaf_sequence == 2
        ));
        let checkout_sequence = records[3].sequence;
        assert_eq!(
            agent.runtime_snapshot_for_test().active_context.branch_id,
            "feature-alpha"
        );
        assert_eq!(
            agent.runtime_snapshot_for_test().context_scope_revision,
            checkout_sequence
        );
    }

    #[test]
    fn create_branch_rejects_active_context_experiment_without_mutation() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-branch-create-experiment-test-{}",
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
        recorder.record_user_message("hello").expect("user message");
        let path = recorder.path().to_path_buf();
        let scope_state = recorder.context_scope_state();
        scope_state.lock().expect("scope lock").active_experiment =
            Some(crate::transcript::ActiveContextExperiment {
                branch_id: "experiment".into(),
                parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                writes_observed: false,
            });
        let transcript = Arc::new(StdMutex::new(recorder));
        let mut agent = test_agent();
        agent.set_context_scope_state(scope_state);
        let agent_snapshot = agent.runtime_snapshot_for_test().clone();
        let records_before = read_records(&path).expect("read records");

        let error = create_context_branch(&mut agent, &transcript, Some("blocked".into()))
            .expect_err("active experiment must reject manual branch creation");

        assert!(error.to_string().contains(
            "manual branch creation is unavailable during an active context experiment; use context__return first"
        ));
        let recorder = transcript.lock().expect("recorder lock");
        assert_eq!(recorder.current_context_branch_id(), None);
        assert_eq!(
            recorder
                .active_context_experiment()
                .expect("active experiment")
                .branch_id,
            "experiment"
        );
        assert_eq!(
            read_records(&path).expect("read records").len(),
            records_before.len()
        );
        assert_eq!(agent.runtime_snapshot_for_test(), &agent_snapshot);
    }

    #[test]
    fn checkout_rebuilds_snapshot_at_selected_branch_tip() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-branch-checkout-test-{}",
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
        recorder
            .record_user_message("root-before")
            .expect("root user");
        recorder
            .record_assistant_message("root-base")
            .expect("root assistant");
        recorder
            .record_context_branch_created(
                "feature",
                crate::transcript::ROOT_CONTEXT_BRANCH_ID,
                3,
                Some("Feature".into()),
            )
            .expect("branch created");
        recorder.set_current_context_branch_id(Some("feature".into()));
        recorder
            .record_user_message("branch-only")
            .expect("branch user");
        recorder.set_current_context_branch_id(None);
        recorder
            .record_assistant_message("root-after")
            .expect("root after");
        let path = recorder.path().to_path_buf();
        let transcript = Arc::new(StdMutex::new(recorder));
        let mut agent = test_agent();

        let snapshot =
            checkout_context_branch(&mut agent, &transcript, "feature").expect("checkout branch");

        assert_eq!(snapshot.branch_id, "feature");
        assert_eq!(snapshot.leaf_sequence, 5);
        let records = read_records(&path).expect("read transcript");
        let checkout_sequence = records.last().expect("checkout record").sequence;
        assert_eq!(snapshot.snapshot.context_scope_revision, checkout_sequence);
        assert_eq!(
            snapshot
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 5]
        );
        assert_eq!(
            transcript
                .lock()
                .expect("recorder lock")
                .current_context_branch_id(),
            Some("feature")
        );
        assert!(matches!(
            records.last().map(|record| &record.event),
            Some(TranscriptEvent::ContextCheckout { branch_id, leaf_sequence })
                if branch_id == "feature" && *leaf_sequence == 5
        ));
    }

    #[test]
    fn checkout_rejects_active_context_experiment_without_mutation() {
        let sessions_dir = std::env::temp_dir().join(format!(
            "letcode-branch-checkout-experiment-test-{}",
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
        recorder.record_user_message("hello").expect("user message");
        let path = recorder.path().to_path_buf();
        let scope_state = recorder.context_scope_state();
        scope_state.lock().expect("scope lock").active_experiment =
            Some(crate::transcript::ActiveContextExperiment {
                branch_id: "experiment".into(),
                parent_branch_id: crate::transcript::ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 2,
                writes_observed: false,
            });
        let transcript = Arc::new(StdMutex::new(recorder));
        let mut agent = test_agent();
        agent.set_context_scope_state(scope_state);
        let agent_snapshot = agent.runtime_snapshot_for_test().clone();
        let records_before = read_records(&path).expect("read records");

        let error = checkout_context_branch(&mut agent, &transcript, "main")
            .expect_err("active experiment must reject manual checkout");

        assert!(error.to_string().contains(
            "manual branch checkout is unavailable during an active context experiment; use context__return first"
        ));
        let recorder = transcript.lock().expect("recorder lock");
        assert_eq!(recorder.current_context_branch_id(), None);
        assert_eq!(
            recorder
                .active_context_experiment()
                .expect("active experiment")
                .branch_id,
            "experiment"
        );
        assert_eq!(
            read_records(&path).expect("read records").len(),
            records_before.len()
        );
        assert_eq!(agent.runtime_snapshot_for_test(), &agent_snapshot);
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
        let snapshot = transcript_projection::project_session_restore_snapshot(
            session_id.clone(),
            records,
            None,
        )
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
        assert_eq!(runtime.state().footer_status.summary, "Compacting context");
        assert_eq!(runtime.state().phase, AppPhase::Running);
    }

    #[test]
    fn compact_noop_notice_remains_visible_after_done() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::new(
            "Nothing to compact yet",
        )));
        runtime.apply_runner_event(RunnerEvent::Status("Nothing to compact yet".into()));
        runtime.apply_runner_event(RunnerEvent::Done);

        assert!(runtime.state().timeline.items().iter().any(|item| matches!(
            item,
            crate::tui::TimelineItem::Notice(notice)
                if notice.message == "Nothing to compact yet"
        )));
        assert_eq!(runtime.state().phase, AppPhase::Completed);
    }

    #[test]
    fn auto_compaction_stream_events_render_like_manual_compact() {
        let mut runtime = runtime();

        runtime.apply_runner_event(RunnerEvent::UserMessage(UserMessageEvent::new("hello")));
        assert_eq!(
            runtime.state().footer_status.summary,
            "Waiting for assistant"
        );

        runtime.apply_runner_event(RunnerEvent::Status("Compacting context".into()));
        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::new(compaction_separator(
            COMPACTION_SEPARATOR_LABEL,
        ))));
        runtime.apply_runner_event(RunnerEvent::AssistantDelta(
            AssistantDeltaEvent::with_message_id(COMPACTION_MESSAGE_ID, "summary part"),
        ));
        runtime.apply_runner_event(RunnerEvent::AssistantDone {
            message_id: Some(COMPACTION_MESSAGE_ID.into()),
        });
        runtime.apply_runner_event(RunnerEvent::Notice(NoticeEvent::new(compaction_separator(
            COMPACTION_SEPARATOR_LABEL,
        ))));
        runtime.apply_runner_event(RunnerEvent::Status(
            "Context compacted (3 history items retained)".into(),
        ));

        assert_ne!(
            runtime.state().footer_status.summary,
            "Waiting for assistant"
        );
        assert_eq!(
            runtime.state().footer_status.summary,
            "Context compacted (3 history items retained)"
        );
        assert!(runtime.state().timeline.items().iter().any(|item| matches!(
            item,
            crate::tui::TimelineItem::Notice(notice)
                if notice.message == compaction_separator(COMPACTION_SEPARATOR_LABEL)
        )));
        assert!(runtime.state().timeline.items().iter().any(|item| matches!(
            item,
            crate::tui::TimelineItem::Assistant(message)
                if message.id.as_deref() == Some(COMPACTION_MESSAGE_ID)
                    && message.text == "summary part"
                    && !message.streaming
        )));
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
        assert_eq!(
            runtime.state().footer_status.detail.as_deref(),
            Some("expanded")
        );

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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Transcript scrollbar"
        );
        assert_eq!(
            runtime.state().footer_status.detail.as_deref(),
            Some("hidden")
        );

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
            ("/child", ChildNavigation::First),
            ("/children next", ChildNavigation::Next),
            ("/child prev", ChildNavigation::Prev),
            ("/parent", ChildNavigation::First),
        ] {
            let mut runtime = runtime();
            runtime.state_mut().set_input(input);

            let command = runtime
                .handle_input_action(InputAction::Submit)
                .expect("command succeeds");

            let expected = if input == "/parent" {
                Some(RuntimeCommand::ViewParent)
            } else {
                Some(RuntimeCommand::ViewChild(expected))
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
        );
        runtime.apply_runner_event(RunnerEvent::ChildAppEvent {
            child_session_id: child_session_id.clone(),
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

        let active_child = crate::transcript::ChildSessionSummary {
            parent_session_id: parent_session_id.clone(),
            parent_run_id: "turn-1".into(),
            child_session_id: child_session_id.clone(),
            agent_name: "explorer".into(),
            status: "running".into(),
            summary: "still running".into(),
            timestamp_ms: 1,
        };

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            Some(active_child),
            ChildNavigation::First,
            None,
        )
        .expect("send child view succeeds");

        assert_eq!(selected.as_deref(), Some(child_session_id.as_str()));
        match rx.try_recv().expect("view event") {
            RunnerEvent::ChildSessionViewed { total, .. } => assert_eq!(total, 1),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn child_navigation_resorts_active_child_fallback() {
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

        let active_child = crate::transcript::ChildSessionSummary {
            parent_session_id: parent_session_id.clone(),
            parent_run_id: "turn-2".into(),
            child_session_id: active_child_session_id.clone(),
            agent_name: "fixer".into(),
            status: "running".into(),
            summary: "still running".into(),
            timestamp_ms: 1,
        };

        let transcript = Arc::new(StdMutex::new(parent));
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            Some(active_child),
            ChildNavigation::First,
            None,
        )
        .expect("send child view succeeds");

        assert_eq!(selected.as_deref(), Some(active_child_session_id.as_str()));
        match rx.try_recv().expect("view event") {
            RunnerEvent::ChildSessionViewed {
                child_session_id,
                index,
                total,
                ..
            } => {
                assert_eq!(child_session_id, active_child_session_id);
                assert_eq!(index, 0);
                assert_eq!(total, 2);
            }
            other => panic!("unexpected event: {other:?}"),
        }
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

        let selected = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            None,
            ChildNavigation::First,
            Some(second_id.as_str()),
        )
        .expect("send child view succeeds");
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
            Some(RuntimeCommand::ViewChild(ChildNavigation::First))
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

        let next = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            None,
            ChildNavigation::Next,
            Some(first_id.as_str()),
        )
        .expect("next succeeds");
        assert_eq!(next.as_deref(), Some(second_id.as_str()));
        let _ = rx.try_recv();

        let prev = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            None,
            ChildNavigation::Prev,
            Some(second_id.as_str()),
        )
        .expect("prev succeeds");
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

        let next = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            None,
            ChildNavigation::Next,
            None,
        )
        .expect("next succeeds");
        assert_eq!(next.as_deref(), Some(first_id.as_str()));
        let _ = rx.try_recv();

        let prev = send_child_session_view(
            &tx,
            &sessions_dir,
            &transcript,
            None,
            ChildNavigation::Prev,
            None,
        )
        .expect("prev succeeds");
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
        assert_eq!(
            runtime.state().footer_status.summary,
            "Unknown child navigation: sideways. Use first, next, or prev."
        );
    }

    #[test]
    fn bare_delegate_without_task_shows_usage() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@fixer");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Usage: @fixer <task>"
        );
    }

    #[test]
    fn delegate_explorer_routes_to_runtime_command() {
        let mut runtime = runtime();
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
        assert_eq!(runtime.state().footer_status.summary, "Starting explorer");
        assert!(matches!(
            runtime.state().timeline.items().last(),
            Some(TimelineItem::Delegation(item))
                if item.agent_name == "explorer" && item.task == "inspect src/agent.rs"
        ));
    }

    #[test]
    fn unknown_delegate_shows_error() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("@unknown foo");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Unknown expert: @unknown. Use @explorer, @fixer, @oracle, @designer, @librarian, or @general."
        );
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
        assert_eq!(runtime.state().footer_status.summary, "Starting fixer");
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
        let (_tx, rx) = mpsc::unbounded_channel();
        let mut runtime = TuiRuntime::new(
            TuiState::new("gpt-5.5", "GPT-5.5", "default"),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
            std::env::temp_dir(),
        );
        let mut dialog = DialogState::new(
            DialogKind::ReasoningPicker,
            "Reasoning effort",
            None,
            reasoning_dialog_items(),
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
    fn slash_model_updates_state_and_runner_command() {
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
            .push_notice("current session notice");

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
        assert_eq!(runtime.state().footer_status.summary, "Session resumed");
        assert!(runtime.state().active_session);
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
            .push_notice("current session notice");

        runtime.apply_runner_event(RunnerEvent::SessionStarted {
            session_id: "new-session".into(),
            records: Vec::new(),
            runtime_context: event_context("new-session", 1),
        });

        assert_eq!(runtime.state().footer_status.summary, "New session started");
        assert_eq!(runtime.state().timeline.items().len(), 0);
        assert!(!runtime.state().active_session);
        assert!(runtime.state().show_dashboard());
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

        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: handle.clone(),
        });

        assert_eq!(runtime.state().phase, AppPhase::WaitingForPermission);
        assert!(runtime.pending_permission_handle().is_some());

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
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle,
        });
        runtime.apply_runner_event(RunnerEvent::ChildSessionViewed {
            parent_session_id: "parent-session".into(),
            child_session_id: "child-session".into(),
            agent_name: "explorer".into(),
            index: 0,
            total: 1,
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
            crate::tui::PermissionResponse::Approve
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
            event: PermissionRequestEvent::new("call-1", "shell__exec", "cargo test"),
            handle: RunnerPermissionRequest::new(tx),
        });
        runtime.state_mut().restore_parent_timeline_view();

        runtime
            .handle_input_action(InputAction::ApprovePermission)
            .expect("approve succeeds");

        assert_eq!(
            rx.await.expect("approval received"),
            PermissionResponse::Approve
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
        assert_eq!(runtime.state().footer_status.summary, "Permission denied");
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
}
