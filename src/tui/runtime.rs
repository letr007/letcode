use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::permission::PermissionMode;
use crate::transcript::TranscriptRecorder;

use super::events::{AppEvent, ErrorEvent};
use super::input::{InputAction, apply_edit_action, map_key_event};
use super::render;
use super::runner::{AgentRunner, RunnerEvent, RunnerPermissionRequest};
use super::state::TuiState;
use super::terminal::OwnedTerminal;
use async_openai::config::Config;
use std::sync::{Arc, Mutex as StdMutex};

const PAGE_SCROLL_ROWS: u16 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCommand {
    SubmitPrompt(String),
    SetPermissionMode(PermissionMode),
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
}

impl TuiRuntime {
    pub fn new(state: TuiState, runner_rx: mpsc::UnboundedReceiver<RunnerEvent>) -> Self {
        Self {
            state,
            runner_rx,
            pending_permission_handle: None,
            submitted_prompts: Vec::new(),
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
                    "Commands: /help, /exit, /quit, /permission, /permission safe, /permission default, /permission solo --yes",
                );
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            "/permission" | "/perm" => self.handle_permission_command(&parts),
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
                self.push_command_notice(format!(
                    "permission mode: {} · available: safe, default, solo",
                    self.state.permission_mode_label
                ));
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            Some("safe") => Ok(Some(self.set_permission_mode_command(PermissionMode::Safe))),
            Some("default") => Ok(Some(
                self.set_permission_mode_command(PermissionMode::Default),
            )),
            Some("solo") if parts.get(2).copied() == Some("--yes") => {
                Ok(Some(self.set_permission_mode_command(PermissionMode::Solo)))
            }
            Some("solo") => {
                self.push_command_notice(
                    "solo mode allows write and command tools without asking. Use /permission solo --yes to enable it explicitly.",
                );
                Ok(Some(SubmittedCommand::LocalOnly))
            }
            Some(other) => {
                self.push_command_notice(format!(
                    "Unknown permission mode: {other}. Use safe, default, or solo --yes."
                ));
                Ok(Some(SubmittedCommand::LocalOnly))
            }
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
}

enum RunnerCommand {
    Prompt(String),
    SetPermissionMode(PermissionMode),
}

pub async fn run_tui<C>(
    agent: Agent<C>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    api_key_configured: bool,
) -> Result<()>
where
    C: Config + Send + 'static,
{
    let model_label = agent.model().to_string();
    let permission_mode_label = agent.permission_mode().to_string();
    let mut state = TuiState::new(model_label, permission_mode_label);

    if !api_key_configured {
        state.timeline.push_notice(
            "OPENAI_API_KEY is not set. The TUI is available, but prompt submissions will return an error until the key is configured.",
        );
        state.set_footer(
            "Missing OPENAI_API_KEY",
            Some("Prompt submission is disabled until the API key is set".into()),
        );
    }

    let (runner_tx, runner_rx) = mpsc::unbounded_channel();
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<RunnerCommand>();
    let mut runtime = TuiRuntime::new(state, runner_rx);
    let mut terminal = OwnedTerminal::new()?;
    let mut drawer = TerminalDrawer::new(&mut terminal);

    let runner_task = tokio::spawn(async move {
        let runner = AgentRunner::with_transcript(runner_tx.clone(), transcript.clone());
        let mut agent = agent;

        while let Some(command) = prompt_rx.recv().await {
            let prompt = match command {
                RunnerCommand::Prompt(prompt) => prompt,
                RunnerCommand::SetPermissionMode(mode) => {
                    agent.set_permission_mode(mode);
                    continue;
                }
            };

            if !api_key_configured {
                let _ = runner_tx.send(RunnerEvent::Error(ErrorEvent::new(
                    "OPENAI_API_KEY is not set. Set it and restart letcode before sending model requests.",
                )));
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
    use crate::tui::{
        AppPhase, PermissionDecision, PermissionRequestEvent, PermissionResolutionEvent,
        RunnerEvent, RunnerPermissionRequest, UserMessageEvent,
    };
    use tokio::sync::{mpsc, oneshot};

    fn runtime() -> TuiRuntime {
        let (_tx, rx) = mpsc::unbounded_channel();
        TuiRuntime::new(TuiState::default(), rx)
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
    fn slash_permission_updates_state_and_runner_command() {
        let mut runtime = runtime();
        runtime.state_mut().set_input("/permission safe");

        let command = runtime
            .handle_input_action(InputAction::Submit)
            .expect("command succeeds");

        assert_eq!(
            command,
            Some(RuntimeCommand::SetPermissionMode(PermissionMode::Safe))
        );
        assert_eq!(runtime.state().permission_mode_label, "safe");
        assert!(runtime.submitted_prompts().is_empty());
    }

    #[test]
    fn runner_permission_events_update_state_and_handle() {
        let mut runtime = runtime();
        let (tx, _rx) = oneshot::channel();
        let handle = RunnerPermissionRequest::new(tx);

        runtime.apply_runner_event(RunnerEvent::PermissionRequested {
            event: PermissionRequestEvent::new("call-1", "bash", "cargo test"),
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
                PermissionRequestEvent::new("call-a", "bash", "ls"),
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
                PermissionRequestEvent::new("call-b", "bash", "rm"),
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
        let mut runtime = TuiRuntime::new(TuiState::default(), rx);
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
}
