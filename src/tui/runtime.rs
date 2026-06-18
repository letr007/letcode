use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::command::{
    ChildNavigation as SharedChildNavigation, CommandIntent, ToolOutputMode, help_summary,
    parse_command,
};
use crate::mcp;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::subagent::SubagentRuntime;
use crate::tool::ToolHandler;
use crate::transcript::{
    SessionSummary, TranscriptRecorder, has_session_content, list_child_sessions_for_parent,
    list_sessions, read_child_session_records, remove_empty_session_file,
};

use super::events::{AppEvent, ErrorEvent};
use super::input::{InputAction, apply_edit_action, map_key_event, map_mouse_event};
use super::preferences::TuiPreferences;
use super::render;
use super::runner::{AgentRunner, RunnerEvent, RunnerPermissionRequest};
use super::slash::{SlashCommandEntry, matching_slash_commands};
use super::state::{DialogItem, DialogKind, DialogState, TuiState};
use super::terminal::OwnedTerminal;
use async_openai::config::Config;
use std::sync::{Arc, Mutex as StdMutex};

const PAGE_SCROLL_ROWS: u16 = 10;
const CHILD_NAVIGATION_PREFIX_TIMEOUT_TICKS: u8 = 20;

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
    SubmitPrompt(String),
    Explore(String),
    Fixer(String),
    ViewChild(ChildNavigation),
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    Interrupt,
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
    pending_permission_handle: Option<RunnerPermissionRequest>,
    interrupt_confirmation_pending: bool,
    submitted_prompts: Vec<String>,
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
            pending_permission_handle: None,
            interrupt_confirmation_pending: false,
            submitted_prompts: Vec::new(),
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

    pub fn submitted_prompts(&self) -> &[String] {
        &self.submitted_prompts
    }

    pub fn pending_permission_handle(&self) -> Option<&RunnerPermissionRequest> {
        self.pending_permission_handle.as_ref()
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
        match &event {
            RunnerEvent::PermissionRequested { handle, .. } => {
                self.pending_permission_handle = Some(handle.clone());
            }
            RunnerEvent::PermissionResolved(_) => {
                self.pending_permission_handle = None;
            }
            RunnerEvent::Error(_) | RunnerEvent::Done | RunnerEvent::Interrupted => {
                self.pending_permission_handle = None;
                self.interrupt_confirmation_pending = false;
            }
            RunnerEvent::SessionResumed {
                session_id,
                messages,
                records,
                evidence_count,
            } => {
                self.pending_permission_handle = None;
                let message_count = messages.len();
                self.state.replace_session_timeline_from_records(records);
                self.state.set_footer(
                    "Session resumed",
                    Some(format!(
                        "{} ({} messages, {} evidence)",
                        session_id, message_count, evidence_count
                    )),
                );
            }
            RunnerEvent::ChildSessionViewed {
                parent_session_id,
                child_session_id,
                agent_name,
                index,
                total,
                records,
            } => {
                self.pending_permission_handle = None;
                self.state.replace_child_timeline_from_records(
                    records,
                    parent_session_id.clone(),
                    child_session_id.clone(),
                    agent_name.clone(),
                    *index,
                    *total,
                );
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
            RunnerEvent::SessionStarted { session_id } => {
                self.pending_permission_handle = None;
                self.state.replace_session_timeline(Vec::new());
                self.state
                    .set_footer("New session started", Some(session_id.clone()));
            }
            RunnerEvent::Status(message) => {
                self.state.set_footer(message.clone(), None);
            }
            _ => {}
        }

        if let Some(app_event) = event.app_event() {
            self.state.apply_event(app_event);
        }
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
            InputAction::DialogNext => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.select_next();
                }
                Ok(None)
            }
            InputAction::DialogPrev => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.select_previous();
                }
                Ok(None)
            }
            InputAction::DialogAccept => self.handle_dialog_accept(),
            InputAction::DialogCancel => {
                self.state.close_dialog();
                self.state.set_footer("Dialog closed", None);
                Ok(None)
            }
            InputAction::DialogInsert(ch) => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.insert_query_char(ch);
                }
                Ok(None)
            }
            InputAction::DialogBackspace => {
                if let Some(dialog) = self.state.dialog_mut() {
                    dialog.pop_query_char();
                }
                Ok(None)
            }
            InputAction::Submit => self.handle_submit(),
            InputAction::ApprovePermission => {
                if let Some(handle) = self.pending_permission_handle.take() {
                    handle.approve()?;
                }
                Ok(None)
            }
            InputAction::DenyPermission => {
                if let Some(handle) = self.pending_permission_handle.take() {
                    handle.deny()?;
                }
                Ok(None)
            }
            InputAction::Interrupt => self.handle_interrupt(),
            InputAction::Quit => {
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
                    self.state.set_footer(
                        "Failed to refresh child transcript",
                        Some(error.to_string()),
                    );
                }
                self.state.apply_event(AppEvent::Tick);
                Ok(None)
            }
            InputAction::Backspace | InputAction::Insert(_) | InputAction::NoOp => Ok(None),
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

    fn handle_submit(&mut self) -> Result<Option<RuntimeCommand>> {
        if self.state.pending_permission.is_some() {
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

        let prompt = self.state.input_buffer.trim().to_string();
        if prompt.is_empty() {
            return Ok(None);
        }

        let running_navigation = matches!(self.state.phase, super::state::AppPhase::Running)
            && child_view_allows_prompt(&prompt);
        if matches!(self.state.phase, super::state::AppPhase::Running) && !running_navigation {
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

        if let Some(command) = self.handle_command(&prompt)? {
            self.state.clear_input();
            return Ok(match command {
                SubmittedCommand::LocalOnly => None,
                SubmittedCommand::Runtime(command) => Some(command),
            });
        }

        self.state.clear_input();
        self.state.mark_session_active();
        self.state.phase = super::state::AppPhase::Running;
        self.state.set_footer(
            "Submitting prompt",
            Some("Waiting for runner events".into()),
        );
        self.submitted_prompts.push(prompt.clone());

        Ok(Some(RuntimeCommand::SubmitPrompt(prompt)))
    }

    fn handle_interrupt(&mut self) -> Result<Option<RuntimeCommand>> {
        if !matches!(self.state.phase, super::state::AppPhase::Running) {
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

    fn handle_command(&mut self, prompt: &str) -> Result<Option<SubmittedCommand>> {
        match parse_command(prompt) {
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
            Ok(CommandIntent::ResumeShow) => self.show_resume_dialog(),
            Ok(CommandIntent::Resume(session_id)) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ResumeSession(session_id),
            ))),
            Ok(CommandIntent::NewSession) => {
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::NewSession)))
            }
            Ok(CommandIntent::Explore(task)) => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.state
                    .set_footer("Starting explorer", Some(task.clone()));
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Explore(
                    task,
                ))))
            }
            Ok(CommandIntent::Fixer(task)) => {
                self.state.mark_session_active();
                self.state.phase = super::state::AppPhase::Running;
                self.state.set_footer("Starting fixer", Some(task.clone()));
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::Fixer(task))))
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

        self.state.close_dialog();
        match kind {
            DialogKind::ModelPicker => {
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
                let effort = parse_reasoning_effort(&selected.id)
                    .expect("reasoning picker items should use valid effort ids");
                self.state
                    .set_reasoning_effort_label(Some(reasoning_effort_status_label(Some(effort))));
                Ok(Some(RuntimeCommand::SetReasoningEffort(effort)))
            }
            DialogKind::SessionPicker => Ok(Some(RuntimeCommand::ResumeSession(selected.id))),
        }
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
        let matches = matching_slash_commands(&self.state.input_buffer);
        matches
            .get(
                self.state
                    .slash_panel_selected
                    .min(matches.len().saturating_sub(1)),
            )
            .copied()
    }

    fn select_next_slash_command(&mut self) {
        let matches = matching_slash_commands(&self.state.input_buffer);
        if matches.is_empty() {
            self.state.slash_panel_selected = 0;
            return;
        }

        self.state.slash_panel_selected = (self.state.slash_panel_selected + 1) % matches.len();
    }

    fn select_previous_slash_command(&mut self) {
        let matches = matching_slash_commands(&self.state.input_buffer);
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
}

enum RunnerCommand {
    Prompt(String),
    Explore(String),
    Fixer(String),
    ViewChild(ChildNavigation),
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
        .last_user_summary
        .clone()
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
        "/help" | "/?" | "/exit" | "/quit" | "/child" | "/children" | "/parent" | "/tool-output"
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

fn send_parent_session_view(
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<()> {
    let (session_id, records) = current_session_records(transcript)?;
    let messages = crate::transcript::restore_conversation_messages(&records);
    let evidence = crate::transcript::restore_session_evidence(&records)?;
    let _ = runner_tx.send(RunnerEvent::SessionResumed {
        session_id,
        messages,
        records,
        evidence_count: evidence.len(),
    });
    Ok(())
}

fn send_child_session_view(
    runner_tx: &mpsc::UnboundedSender<RunnerEvent>,
    sessions_dir: &std::path::Path,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    active_child: Option<crate::transcript::ChildSessionSummary>,
    navigation: ChildNavigation,
    active_child_session_id: Option<&str>,
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
    }
    if children.is_empty() {
        let _ = runner_tx.send(RunnerEvent::Status(
            "No child subagent transcripts for this session".into(),
        ));
        return Ok(None);
    }

    let current_index = active_child_session_id
        .and_then(|child_session_id| {
            children
                .iter()
                .position(|child| child.child_session_id == child_session_id)
        })
        .or_else(|| {
            active_child.as_ref().and_then(|active_child| {
                children
                    .iter()
                    .position(|child| child.child_session_id == active_child.child_session_id)
            })
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
            .unwrap_or(0),
    };
    let child = &children[index];
    let records = read_child_session_records(sessions_dir, &child.child_session_id)?;
    let _ = runner_tx.send(RunnerEvent::ChildSessionViewed {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        records,
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

    let records = read_child_session_records(sessions_dir, &metadata.child_session_id)?;
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

    if completed_position.is_some() {
        state.replace_child_timeline_from_records(
            &records,
            metadata.parent_session_id,
            metadata.child_session_id,
            metadata.agent_name,
            next_index,
            next_total,
        );
    } else {
        state.refresh_child_timeline_from_records(&records);
    }
    Ok(true)
}

pub async fn run_tui<C>(
    agent: Agent<C>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    preferences_dir: PathBuf,
    api_key_configured: bool,
    api_key_hint: String,
    provider_label: String,
    available_models: Vec<AvailableModel>,
    mcp_tools_rx: Option<mpsc::UnboundedReceiver<anyhow::Result<Vec<mcp::McpTool>>>>,
) -> Result<()>
where
    C: Config + Clone + Send + Sync + 'static,
{
    let model_id = agent.model().to_string();
    let model_label = agent.model().to_string();
    let permission_mode_label = agent.permission_mode().to_string();
    let mut state = TuiState::new(model_id, model_label, permission_mode_label);
    let preferences = TuiPreferences::load_from_dir(&preferences_dir);
    state.set_tool_output_expanded(preferences.tool_output_expanded);
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

    let (runner_tx, runner_rx) = mpsc::unbounded_channel();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<RunnerCommand>();
    let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel::<()>();
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
        let mut active_child_session_id: Option<String> = None;

        loop {
            tokio::select! {
                command = prompt_rx.recv() => {
                    let Some(command) = command else {
                        break;
                    };

                    let prompt = match command {
                        RunnerCommand::Prompt(prompt) => prompt,
                        RunnerCommand::Explore(task) => {
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

                            let explore = subagent_runtime.run_explorer(
                                &agent,
                                task,
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
                                Some(runner_tx.clone()),
                            );

                            tokio::pin!(explore);

                            loop {
                                tokio::select! {
                                    biased;
                                    result = &mut explore => {
                                        match result {
                                            Ok(_) => {
                                                let _ = runner_tx.send(RunnerEvent::Done);
                                            }
                                            Err(error) => {
                                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(error.to_string())));
                                                let _ = runner_tx.send(RunnerEvent::Done);
                                            }
                                        }
                                        break;
                                    }
                                    Some(()) = cancel_rx.recv() => {
                                        subagent_runtime.cancel_active();
                                        if let Err(error) = explore.await {
                                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(error.to_string())));
                                        }
                                        let _ = runner_tx.send(RunnerEvent::Interrupted);
                                        break;
                                    }
                                    command = prompt_rx.recv() => {
                                        match command {
                                            Some(RunnerCommand::ViewChild(navigation)) => {
                                                match send_child_session_view(
                                                    &runner_tx,
                                                    &sessions_dir,
                                                    &transcript,
                                                    subagent_runtime.active_child(),
                                                    navigation,
                                                    active_child_session_id.as_deref(),
                                                ) {
                                                    Ok(session_id) => active_child_session_id = session_id,
                                                    Err(error) => {
                                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                            "failed to view child transcript: {error}"
                                                        ))));
                                                    }
                                                }
                                            }
                                            Some(RunnerCommand::ViewParent) => {
                                                active_child_session_id = None;
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
                            continue;
                        }
                        RunnerCommand::Fixer(task) => {
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

                            let fixer = subagent_runtime.run_fixer(
                                &agent,
                                task,
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
                                Some(runner_tx.clone()),
                            );

                            tokio::pin!(fixer);

                            loop {
                                tokio::select! {
                                    biased;
                                    result = &mut fixer => {
                                        match result {
                                            Ok(_) => {
                                                let _ = runner_tx.send(RunnerEvent::Done);
                                            }
                                            Err(error) => {
                                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(error.to_string())));
                                                let _ = runner_tx.send(RunnerEvent::Done);
                                            }
                                        }
                                        break;
                                    }
                                    Some(()) = cancel_rx.recv() => {
                                        subagent_runtime.cancel_active();
                                        if let Err(error) = fixer.await {
                                            let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(error.to_string())));
                                        }
                                        let _ = runner_tx.send(RunnerEvent::Interrupted);
                                        break;
                                    }
                                    command = prompt_rx.recv() => {
                                        match command {
                                            Some(RunnerCommand::ViewChild(navigation)) => {
                                                match send_child_session_view(
                                                    &runner_tx,
                                                    &sessions_dir,
                                                    &transcript,
                                                    subagent_runtime.active_child(),
                                                    navigation,
                                                    active_child_session_id.as_deref(),
                                                ) {
                                                    Ok(session_id) => active_child_session_id = session_id,
                                                    Err(error) => {
                                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                            "failed to view child transcript: {error}"
                                                        ))));
                                                    }
                                                }
                                            }
                                            Some(RunnerCommand::ViewParent) => {
                                                active_child_session_id = None;
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
                            continue;
                        }
                        RunnerCommand::ViewChild(navigation) => {
                            match send_child_session_view(
                                &runner_tx,
                                &sessions_dir,
                                &transcript,
                                subagent_runtime.active_child(),
                                navigation,
                                active_child_session_id.as_deref(),
                            ) {
                                Ok(session_id) => active_child_session_id = session_id,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to view child transcript: {error}"
                                    ))));
                                }
                            }
                            continue;
                        }
                        RunnerCommand::ViewParent => {
                            active_child_session_id = None;
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
                            let messages = crate::transcript::restore_conversation_messages(&records);
                            let evidence = match crate::transcript::restore_session_evidence(&records) {
                                Ok(evidence) => evidence,
                                Err(error) => {
                                    let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore session evidence: {error}"
                                    ))));
                                    continue;
                                }
                            };
                            let max_turn_id = crate::transcript::restore_max_turn_id(&records);
                            if let Err(error) = agent.restore_session_context(
                                messages.clone(),
                                evidence.clone(),
                                max_turn_id,
                            ) {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to restore session context: {error}"
                                ))));
                                continue;
                            }
                            let new_recorder =
                                match TranscriptRecorder::open_existing(&sessions_dir, &session_id) {
                                    Ok(recorder) => recorder,
                                    Err(error) => {
                                        let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                            format!("failed to open session transcript: {error}"),
                                        )));
                                        continue;
                                    }
                                };
                            let old_empty_session_path = transcript
                                .lock()
                                .ok()
                                .and_then(|recorder| empty_session_path(recorder.path()));
                            if let Ok(mut recorder) = transcript.lock() {
                                *recorder = new_recorder;
                            } else {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                    "transcript recorder poisoned",
                                )));
                                continue;
                            }
                            if let Some(path) = old_empty_session_path
                                && path != sessions_dir.join(format!("{session_id}.jsonl"))
                            {
                                let _ = std::fs::remove_file(path);
                            }
                            active_child_session_id = None;
                            let _ = runner_tx.send(RunnerEvent::SessionResumed {
                                session_id,
                                messages,
                                records,
                                evidence_count: evidence.len(),
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

                            if let Err(error) = agent.restore_session_context(Vec::new(), Vec::new(), 0) {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to clear session context: {error}"
                                ))));
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
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                    "failed to record session start: {error}"
                                ))));
                                continue;
                            }
                            let session_id = new_recorder.session_id().to_string();
                            let new_path = new_recorder.path().to_path_buf();
                            let old_empty_session_path = transcript
                                .lock()
                                .ok()
                                .and_then(|recorder| empty_session_path(recorder.path()));
                            if let Ok(mut recorder) = transcript.lock() {
                                *recorder = new_recorder;
                            } else {
                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                                    "transcript recorder poisoned",
                                )));
                                continue;
                            }
                            if let Some(path) = old_empty_session_path
                                && path != new_path
                            {
                                let _ = std::fs::remove_file(path);
                            }
                            active_child_session_id = None;
                            let _ = runner_tx.send(RunnerEvent::SessionStarted { session_id });
                            continue;
                        }
                    };

                    if !api_key_configured {
                        send_missing_api_key_error(&runner_tx, &api_key_hint);
                        continue;
                    }

                    let run = runner.run_prompt(&mut agent, prompt);
                    tokio::pin!(run);

                    loop {
                        tokio::select! {
                            _ = &mut run => break,
                            command = prompt_rx.recv() => {
                                match command {
                                    Some(RunnerCommand::ViewChild(navigation)) => {
                                        match send_child_session_view(
                                            &runner_tx,
                                            &sessions_dir,
                                            &transcript,
                                            subagent_runtime.active_child(),
                                            navigation,
                                            active_child_session_id.as_deref(),
                                        ) {
                                            Ok(session_id) => active_child_session_id = session_id,
                                            Err(error) => {
                                                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                                                    "failed to view child transcript: {error}"
                                                ))));
                                            }
                                        }
                                    }
                                    Some(RunnerCommand::ViewParent) => {
                                        active_child_session_id = None;
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
                            Some(()) = cancel_rx.recv() => {
                                subagent_runtime.cancel_active();
                                let _ = runner_tx.send(RunnerEvent::Interrupted);
                                break;
                            }
                        }
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
        runtime.draw(&mut drawer)?;

        if runtime.state().quit_requested {
            break;
        }

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    let action = map_key_event(runtime.state(), key);
                    if let Some(command) = runtime.handle_input_action(action)? {
                        match command {
                            RuntimeCommand::SubmitPrompt(prompt) => {
                                if prompt_tx.send(RunnerCommand::Prompt(prompt)).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::Explore(task) => {
                                if prompt_tx.send(RunnerCommand::Explore(task)).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::Fixer(task) => {
                                if prompt_tx.send(RunnerCommand::Fixer(task)).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::ViewChild(navigation) => {
                                if prompt_tx
                                    .send(RunnerCommand::ViewChild(navigation))
                                    .is_err()
                                {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::ViewParent => {
                                if prompt_tx.send(RunnerCommand::ViewParent).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::SetPermissionMode(mode) => {
                                if prompt_tx
                                    .send(RunnerCommand::SetPermissionMode(mode))
                                    .is_err()
                                {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::SetModel(model) => {
                                if prompt_tx.send(RunnerCommand::SetModel(model)).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::SetReasoningEffort(effort) => {
                                if prompt_tx
                                    .send(RunnerCommand::SetReasoningEffort(effort))
                                    .is_err()
                                {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::ResumeSession(session_id) => {
                                if prompt_tx
                                    .send(RunnerCommand::ResumeSession(session_id))
                                    .is_err()
                                {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::NewSession => {
                                if prompt_tx.send(RunnerCommand::NewSession).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::Interrupt => {
                                if cancel_tx.send(()).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    let action = map_mouse_event(runtime.state(), mouse);
                    if let Some(command) = runtime.handle_input_action(action)? {
                        match command {
                            RuntimeCommand::ViewChild(navigation) => {
                                if prompt_tx
                                    .send(RunnerCommand::ViewChild(navigation))
                                    .is_err()
                                {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::ViewParent => {
                                if prompt_tx.send(RunnerCommand::ViewParent).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::Interrupt => {
                                if cancel_tx.send(()).is_err() {
                                    runtime.apply_runner_event(RunnerEvent::Error(
                                        ErrorEvent::new("TUI runner task is no longer available"),
                                    ));
                                    runtime.apply_runner_event(RunnerEvent::Done);
                                }
                            }
                            RuntimeCommand::SubmitPrompt(_)
                            | RuntimeCommand::Explore(_)
                            | RuntimeCommand::Fixer(_)
                            | RuntimeCommand::SetPermissionMode(_)
                            | RuntimeCommand::SetModel(_)
                            | RuntimeCommand::SetReasoningEffort(_)
                            | RuntimeCommand::ResumeSession(_)
                            | RuntimeCommand::NewSession => {}
                        }
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
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::tui::{
        AppPhase, PermissionDecision, PermissionRequestEvent, PermissionResolutionEvent,
        RunnerEvent, RunnerPermissionRequest, UserMessageEvent,
    };
    use tokio::sync::{mpsc, oneshot};

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
    fn interrupted_runner_event_returns_to_prompt_ready_state() {
        let mut runtime = runtime();
        runtime.state.phase = AppPhase::Running;

        runtime.apply_runner_event(RunnerEvent::Interrupted);

        assert_eq!(runtime.state().phase, AppPhase::Completed);
        assert_eq!(runtime.state().footer_status.summary, "Interrupted");
        assert!(runtime.state().pending_permission.is_none());
    }

    #[test]
    fn child_transcript_view_blocks_parent_mutating_submit_paths() {
        for input in [
            "ask the parent agent",
            "/explore inspect src/agent.rs",
            "/fixer wire agent__fixer tool",
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
                event: TranscriptEvent::UserMessage {
                    content: "parent prompt".into(),
                },
            }]);
        runtime.state_mut().replace_child_timeline_from_records(
            &[TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 1,
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
            "Commands: /help, /exit, /quit, /model, /reasoning, /permission, /tool-output, /resume, /new, /explore, /fixer, /child, /parent"
        );
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
    fn slash_explore_without_task_shows_usage() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/explore");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Usage: /explore <task>"
        );
    }

    #[test]
    fn slash_explore_with_task_routes_to_runtime_command() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_input("/explore inspect src/agent.rs");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::Explore("inspect src/agent.rs".into()))
        );
        assert_eq!(runtime.state().footer_status.summary, "Starting explorer");
    }

    #[test]
    fn slash_fixer_without_task_shows_usage() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/fixer");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert_eq!(
            runtime.state().footer_status.summary,
            "Usage: /fixer <task>"
        );
    }

    #[test]
    fn slash_fixer_with_task_routes_to_runtime_command() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .set_input("/fixer wire agent__fixer tool");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::Fixer("wire agent__fixer tool".into()))
        );
        assert_eq!(runtime.state().footer_status.summary, "Starting fixer");
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
    fn session_resumed_event_replaces_timeline_not_appends() {
        let mut runtime = runtime();
        runtime
            .state_mut()
            .timeline
            .push_notice("current session notice");

        runtime.apply_runner_event(RunnerEvent::SessionResumed {
            session_id: "session-1".into(),
            messages: vec![crate::agent::ConversationMessage {
                role: crate::agent::ConversationRole::User,
                content: "old prompt".into(),
            }],
            records: vec![crate::transcript::TranscriptRecord {
                session_id: "session-1".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: crate::transcript::TranscriptEvent::UserMessage {
                    content: "old prompt".into(),
                },
            }],
            evidence_count: 2,
        });

        assert!(matches!(
            runtime.state().timeline.items().first(),
            Some(crate::tui::TimelineItem::User(message)) if message.text == "old prompt"
        ));
        assert_eq!(runtime.state().footer_status.summary, "Session resumed");
        assert!(runtime.state().active_session);
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
        });

        assert_eq!(runtime.state().footer_status.summary, "New session started");
        assert_eq!(runtime.state().timeline.items().len(), 0);
        assert!(!runtime.state().active_session);
        assert!(runtime.state().show_dashboard());
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

    #[tokio::test]
    async fn approve_and_deny_actions_respond_through_pending_handle() {
        let mut approve_runtime = runtime();
        let (approve_tx, approve_rx) = oneshot::channel();
        approve_runtime.pending_permission_handle = Some(RunnerPermissionRequest::new(approve_tx));
        approve_runtime.state_mut().pending_permission =
            Some(crate::tui::PermissionView::from_request(
                PermissionRequestEvent::new("call-a", "shell__exec", "ls"),
            ));

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
        deny_runtime.pending_permission_handle = Some(RunnerPermissionRequest::new(deny_tx));
        deny_runtime.state_mut().pending_permission =
            Some(crate::tui::PermissionView::from_request(
                PermissionRequestEvent::new("call-b", "shell__exec", "rm"),
            ));

        deny_runtime
            .handle_input_action(InputAction::DenyPermission)
            .expect("deny succeeds");
        assert_eq!(
            deny_rx.await.expect("denial received"),
            crate::tui::PermissionResponse::Deny
        );
        assert!(deny_runtime.pending_permission_handle().is_none());
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
            messages: Vec::new(),
            records,
            evidence_count: 0,
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
