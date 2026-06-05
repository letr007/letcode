use std::io;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::permission::PermissionMode;
use crate::transcript::{SessionSummary, TranscriptRecorder, has_session_content, list_sessions};

use super::events::{AppEvent, ErrorEvent};
use super::input::{InputAction, apply_edit_action, map_key_event};
use super::render;
use super::runner::{AgentRunner, RunnerEvent, RunnerPermissionRequest};
use super::slash::{SlashCommandEntry, matching_slash_commands};
use super::state::{DialogItem, DialogKind, DialogState, TuiState};
use super::terminal::OwnedTerminal;
use async_openai::config::Config;
use std::sync::{Arc, Mutex as StdMutex};

const PAGE_SCROLL_ROWS: u16 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableModel {
    pub id: String,
    pub label: String,
    pub context_window_tokens: Option<u64>,
}

impl AvailableModel {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            context_window_tokens: None,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCommand {
    SubmitPrompt(String),
    SetPermissionMode(PermissionMode),
    SetModel(String),
    ResumeSession(String),
    NewSession,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SubmittedCommand {
    LocalOnly,
    Runtime(RuntimeCommand),
}

pub trait RuntimeDrawer {
    fn draw(&mut self, state: &TuiState) -> io::Result<()>;
}

#[derive(Debug, Default)]
pub struct NoopDrawer;

impl RuntimeDrawer for NoopDrawer {
    fn draw(&mut self, _state: &TuiState) -> io::Result<()> {
        Ok(())
    }
}

pub struct TuiRuntime {
    state: TuiState,
    runner_rx: mpsc::UnboundedReceiver<RunnerEvent>,
    pending_permission_handle: Option<RunnerPermissionRequest>,
    submitted_prompts: Vec<String>,
    available_models: Vec<AvailableModel>,
    sessions_dir: PathBuf,
}

impl TuiRuntime {
    pub fn new(
        state: TuiState,
        runner_rx: mpsc::UnboundedReceiver<RunnerEvent>,
        available_models: Vec<AvailableModel>,
        sessions_dir: PathBuf,
    ) -> Self {
        Self {
            state,
            runner_rx,
            pending_permission_handle: None,
            submitted_prompts: Vec::new(),
            available_models,
            sessions_dir,
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
            RunnerEvent::Error(_) | RunnerEvent::Done => {
                self.pending_permission_handle = None;
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
                self.state.timeline.push_notice(format!(
                    "resumed session {} ({} messages, {} evidence)",
                    session_id, message_count, evidence_count
                ));
            }
            RunnerEvent::SessionStarted { session_id } => {
                self.pending_permission_handle = None;
                self.state.replace_session_timeline(Vec::new());
                self.state
                    .set_footer("New session started", Some(session_id.clone()));
                self.state
                    .timeline
                    .push_notice(format!("started new session {session_id}"));
            }
            _ => {}
        }

        if let Some(app_event) = event.app_event() {
            self.state.apply_event(app_event);
        }
    }

    pub fn handle_input_action(&mut self, action: InputAction) -> Result<Option<RuntimeCommand>> {
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
            InputAction::Quit => {
                self.state.apply_event(AppEvent::Quit);
                Ok(None)
            }
            InputAction::Tick => {
                self.state.apply_event(AppEvent::Tick);
                Ok(None)
            }
            InputAction::Backspace | InputAction::Insert(_) | InputAction::NoOp => Ok(None),
        }
    }

    pub fn draw<D: RuntimeDrawer>(&self, drawer: &mut D) -> io::Result<()> {
        drawer.draw(&self.state)
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

        if let Some(command) = self.handle_command(&prompt)? {
            self.state.clear_input();
            return Ok(match command {
                SubmittedCommand::LocalOnly => None,
                SubmittedCommand::Runtime(command) => Some(command),
            });
        }

        self.state.clear_input();
        self.state.phase = super::state::AppPhase::Running;
        self.state.set_footer(
            "Submitting prompt",
            Some("Waiting for runner events".into()),
        );
        self.submitted_prompts.push(prompt.clone());

        Ok(Some(RuntimeCommand::SubmitPrompt(prompt)))
    }

    fn handle_command(&mut self, prompt: &str) -> Result<Option<SubmittedCommand>> {
        let command = prompt.trim();
        if command.eq_ignore_ascii_case("exit") || command.eq_ignore_ascii_case("quit") {
            self.state.apply_event(AppEvent::Quit);
            return Ok(Some(SubmittedCommand::LocalOnly));
        }

        if !command.starts_with('/') {
            return Ok(None);
        }

        let parts: Vec<&str> = command.split_whitespace().collect();
        let Some(name) = parts.first().copied() else {
            return Ok(None);
        };

        match name {
            "/exit" | "/quit" => {
                self.state.apply_event(AppEvent::Quit);
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            "/help" | "/?" => {
                self.push_command_notice(
                    "Commands: /help, /exit, /quit, /model, /permission, /resume, /new",
                );
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            "/model" => self.handle_model_command(&parts),
            "/permission" | "/perm" => self.handle_permission_command(&parts),
            "/resume" => self.handle_resume_command(&parts),
            "/new" => Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::NewSession))),
            _ => {
                self.push_command_notice(format!(
                    "Unknown command: {name}. Type /help for available TUI commands."
                ));
                Ok(Some(SubmittedCommand::LocalOnly))
            }
        }
    }

    fn handle_permission_command(&mut self, parts: &[&str]) -> Result<Option<SubmittedCommand>> {
        match parts.get(1).copied() {
            None => {
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
            Some("safe") => Ok(Some(self.set_permission_mode_command(PermissionMode::Safe))),
            Some("default") => Ok(Some(
                self.set_permission_mode_command(PermissionMode::Default),
            )),
            Some("solo") => Ok(Some(self.set_permission_mode_command(PermissionMode::Solo))),
            Some(other) => {
                self.push_command_notice(format!(
                    "Unknown permission mode: {other}. Use safe, default, or solo."
                ));
                Ok(Some(SubmittedCommand::LocalOnly))
            }
        }
    }

    fn handle_model_command(&mut self, parts: &[&str]) -> Result<Option<SubmittedCommand>> {
        match parts.get(1).copied() {
            None => {
                let items = self
                    .available_models
                    .iter()
                    .map(|model| {
                        DialogItem::new(
                            model.id.clone(),
                            model.label.clone(),
                            Some(if model.id == self.state.model_id {
                                format!("{} · current", model.id)
                            } else {
                                model.id.clone()
                            }),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut dialog = DialogState::new(
                    DialogKind::ModelPicker,
                    "Switch model",
                    Some("Select a model to use for subsequent prompts".into()),
                    items,
                );
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
            Some(model_id) => {
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
                        "Unknown model: {model_id}. Available models: {available}"
                    ));
                    return Ok(Some(SubmittedCommand::LocalOnly));
                };

                self.state.set_model(model.id.clone(), model.label.clone());
                self.state
                    .set_model_context_window(model.context_window_tokens);
                self.state.set_footer(
                    "Model updated",
                    Some(format!("using {} ({})", model.label, model.id)),
                );
                self.push_command_notice(format!("model set to {} ({})", model.label, model.id));
                Ok(Some(SubmittedCommand::Runtime(RuntimeCommand::SetModel(
                    model.id,
                ))))
            }
        }
    }

    fn handle_resume_command(&mut self, parts: &[&str]) -> Result<Option<SubmittedCommand>> {
        match parts.get(1).copied() {
            Some(session_id) => Ok(Some(SubmittedCommand::Runtime(
                RuntimeCommand::ResumeSession(session_id.to_string()),
            ))),
            None => {
                let sessions = list_sessions(&self.sessions_dir)?;
                if sessions.is_empty() {
                    self.push_command_notice("No previous sessions found");
                    return Ok(Some(SubmittedCommand::LocalOnly));
                }

                let items = sessions.iter().map(session_dialog_item).collect::<Vec<_>>();
                let dialog = DialogState::new(
                    DialogKind::SessionPicker,
                    "Resume session",
                    Some("Select a previous session to switch to".into()),
                    items,
                );
                self.state.open_dialog(dialog);
                self.state.set_footer(
                    "Session dialog",
                    Some("Choose a session and press Enter".into()),
                );
                Ok(Some(SubmittedCommand::LocalOnly))
            }
        }
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
                self.state.set_footer(
                    "Model updated",
                    Some(format!("using {} ({})", selected.label, selected.id)),
                );
                self.push_command_notice(format!(
                    "model set to {} ({})",
                    selected.label, selected.id
                ));
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
                self.push_command_notice(format!("permission mode set to {label}"));
                Ok(Some(RuntimeCommand::SetPermissionMode(mode)))
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
        self.push_command_notice(format!("permission mode set to {label}"));
        SubmittedCommand::Runtime(RuntimeCommand::SetPermissionMode(mode))
    }

    fn push_command_notice(&mut self, message: impl Into<String>) {
        self.state.timeline.push_notice(message);
        self.state.set_footer("Command handled", None);
    }

    fn selected_slash_command(&self) -> Option<&'static SlashCommandEntry> {
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
    SetPermissionMode(PermissionMode),
    SetModel(String),
    ResumeSession(String),
    NewSession,
}

fn session_dialog_item(session: &SessionSummary) -> DialogItem {
    let label = session
        .last_user_summary
        .clone()
        .or_else(|| session.last_assistant_summary.clone())
        .unwrap_or_else(|| "empty session".into());
    let detail = format!(
        "{} · {} records{}",
        session.session_id,
        session.record_count,
        session
            .model
            .as_ref()
            .map(|model| format!(" · {model}"))
            .unwrap_or_default()
    );
    DialogItem::new(session.session_id.clone(), label, Some(detail))
}

fn empty_session_path(path: &std::path::Path) -> Option<PathBuf> {
    let records = crate::transcript::read_records(path).ok()?;
    (!has_session_content(&records)).then(|| path.to_path_buf())
}

pub async fn run_tui<C>(
    agent: Agent<C>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    api_key_configured: bool,
    api_key_hint: String,
    available_models: Vec<AvailableModel>,
) -> Result<()>
where
    C: Config + Send + 'static,
{
    let model_id = agent.model().to_string();
    let model_label = agent.model().to_string();
    let permission_mode_label = agent.permission_mode().to_string();
    let mut state = TuiState::new(model_id, model_label, permission_mode_label);

    if let Some(active_model) = available_models
        .iter()
        .find(|model| model.id == state.model_id)
    {
        state.set_model(active_model.id.clone(), active_model.label.clone());
        state.set_model_context_window(active_model.context_window_tokens);
    }

    if !api_key_configured {
        state.timeline.push_notice(
            format!(
                "API key is not set for the active provider. The TUI is available, but prompt submissions will return an error until the key is configured. {}",
                api_key_hint
            ),
        );
        state.set_footer("Missing API key", Some(api_key_hint.clone()));
    }

    let (runner_tx, runner_rx) = mpsc::unbounded_channel();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<RunnerCommand>();
    let mut runtime = TuiRuntime::new(state, runner_rx, available_models, sessions_dir.clone());
    let mut terminal = OwnedTerminal::new()?;
    let mut drawer = TerminalDrawer::new(&mut terminal);

    let runner_task = tokio::spawn(async move {
        let runner = AgentRunner::with_transcript(runner_tx.clone(), transcript.clone());
        let mut agent = agent;

        while let Some(command) = prompt_rx.recv().await {
            let prompt = match command {
                RunnerCommand::Prompt(prompt) => prompt,
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
                RunnerCommand::ResumeSession(prefix) => {
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
                    if let Err(error) =
                        agent.restore_session_context(messages.clone(), evidence.clone())
                    {
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
                    let _ = runner_tx.send(RunnerEvent::SessionResumed {
                        session_id,
                        messages,
                        records,
                        evidence_count: evidence.len(),
                    });
                    continue;
                }
                RunnerCommand::NewSession => {
                    if let Err(error) = agent.restore_session_context(Vec::new(), Vec::new()) {
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
                    let _ = runner_tx.send(RunnerEvent::SessionStarted { session_id });
                    continue;
                }
            };

            if !api_key_configured {
                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                    "API key is not set for the active provider. {}",
                    api_key_hint
                ))));
                let _ = runner_tx.send(RunnerEvent::Done);
                continue;
            }

            let _ = runner.run_prompt(&mut agent, prompt).await;
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
    runner_task.abort();

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
    fn draw(&mut self, state: &TuiState) -> io::Result<()> {
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
        TuiRuntime::new(
            TuiState::default(),
            rx,
            vec![AvailableModel::new("gpt-5.5", "GPT-5.5")],
            std::env::temp_dir(),
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
    fn slash_help_is_local_notice_not_agent_prompt() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/help");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        assert!(runtime.state().input_buffer.is_empty());
        assert!(runtime.submitted_prompts().is_empty());
        assert_eq!(runtime.state().timeline.items().len(), 1);
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
        );
        runtime.state_mut().set_input("/model");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(command, None);
        let dialog = runtime.state().dialog().expect("dialog should be open");
        assert_eq!(dialog.title, "Switch model");
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
        assert_eq!(runtime.state().timeline.items().len(), 1);
        assert!(matches!(
            runtime.state().timeline.items().first(),
            Some(crate::tui::TimelineItem::Notice(notice)) if notice.message.contains("new-session")
        ));
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
