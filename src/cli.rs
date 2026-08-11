//! Line CLI / REPL frontend.
//!
//! Same boundary as TUI: submit [`SessionCommand`] through
//! [`SessionEngineIngress`], present [`SessionTransportEvent`] on stdout, and
//! answer permission/question handles. No direct agent/runner ownership.

use crate::command::{CommandIntent, ToolOutputMode, command_metadata, parse_command};
use crate::config::AppConfig;
use crate::permission::{PermissionApproval, PermissionMode};
use crate::request_builder::{ModelReasoningEffort, ModelRequestMetadata};
use crate::session::{
    PermissionRequestEvent, SessionCommand, SessionEngine, SessionEngineIngress,
    SessionEngineProjection, SessionTransportEvent, ToolOutcome,
};
use crate::tool::{QuestionRequest, QuestionResponse};
use crate::transcript::list_sessions;
use crate::user_content::{UserMessageContent, UserMessageSubmission};
use anyhow::{Result, anyhow, bail};
use serde_json::json;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Streaming,
    FinalOnly,
}

#[derive(Debug, Clone)]
struct CliView {
    session_id: String,
    model_id: String,
    model_label: String,
    permission_mode_label: String,
    reasoning_effort: Option<ModelReasoningEffort>,
}

impl CliView {
    fn from_projection(
        projection: SessionEngineProjection,
        reasoning_effort: Option<ModelReasoningEffort>,
    ) -> Self {
        Self {
            session_id: projection.session_id,
            model_id: projection.model_id,
            model_label: projection.model_label,
            permission_mode_label: projection.permission_mode_label,
            reasoning_effort,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandWait {
    UntilDone,
    UntilModelChanged,
    UntilFastMode,
    UntilHistoryLoaded,
    UntilSessionResumed,
    UntilSessionStarted,
    UntilPermissionModeChanged,
    UntilReasoningEffortChanged,
}

pub async fn run_repl(
    mut engine: SessionEngine,
    projection: SessionEngineProjection,
    config: &AppConfig,
    sessions_dir: PathBuf,
    initial_reasoning: Option<ModelReasoningEffort>,
) -> Result<()> {
    let ingress = engine.take_ingress();
    let mut events = engine.take_event_egress().into_receiver();
    let mut view = CliView::from_projection(projection, initial_reasoning);

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match parse_repl_command(input) {
            ReplCommand::Exit => break,
            ReplCommand::Empty => continue,
            ReplCommand::Help => print_repl_help(),
            ReplCommand::PermissionShow => {
                println!(
                    "permission mode: {}\navailable modes: safe, default, auto, yolo",
                    view.permission_mode_label
                );
            }
            ReplCommand::PermissionSet(
                mode @ (PermissionMode::Safe | PermissionMode::Default | PermissionMode::Auto),
            ) => {
                let outcome = submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::SetPermissionMode(mode),
                    CommandWait::UntilPermissionModeChanged,
                    OutputMode::Streaming,
                )
                .await?;
                if outcome.error.is_none() {
                    println!("permission mode set to {mode}");
                }
            }
            ReplCommand::PermissionSet(PermissionMode::Yolo) => {
                print!(
                    "YOLO mode allows write and command tools without asking. Enable YOLO mode? [y/N] "
                );
                io::stdout().flush()?;

                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                let confirm = confirm.trim().to_ascii_lowercase();

                if matches!(confirm.as_str(), "y" | "yes") {
                    let outcome = submit_and_wait(
                        &ingress,
                        &mut events,
                        &mut view,
                        SessionCommand::SetPermissionMode(PermissionMode::Yolo),
                        CommandWait::UntilPermissionModeChanged,
                        OutputMode::Streaming,
                    )
                    .await?;
                    if outcome.error.is_none() {
                        println!("permission mode set to yolo");
                    }
                } else {
                    println!("YOLO mode not enabled");
                }
            }
            ReplCommand::ModelShow => {
                println!("current model: {} ({})", view.model_label, view.model_id);
                println!("available models:");
                for (provider_name, provider) in &config.providers {
                    for model_id in provider.models.keys() {
                        let route = crate::config::ModelRoute::new(provider_name, model_id);
                        println!(
                            "  {} ({})",
                            provider.model_label(model_id),
                            route.display_name()
                        );
                    }
                }
            }
            ReplCommand::ToggleFastMode => {
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::ToggleFastMode,
                    CommandWait::UntilFastMode,
                    OutputMode::Streaming,
                )
                .await?;
            }
            ReplCommand::ModelSet(model_id) => {
                let route = parse_model_route(config, &model_id);
                let Ok(provider) = config.resolve_route(&route) else {
                    println!("unknown model: {model_id}");
                    println!("available models:");
                    for (provider_name, provider) in &config.providers {
                        for available_model_id in provider.models.keys() {
                            let available_route =
                                crate::config::ModelRoute::new(provider_name, available_model_id);
                            println!(
                                "  {} ({})",
                                provider.model_label(available_model_id),
                                available_route.display_name()
                            );
                        }
                    }
                    continue;
                };
                let label = provider.model_label(&route.model);
                let route_display_name = route.display_name();
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::SetModel(route_display_name.clone()),
                    CommandWait::UntilModelChanged,
                    OutputMode::Streaming,
                )
                .await?;
                if view.model_id == route_display_name {
                    view.model_label = label.clone();
                    println!("model set to {} ({})", label, route_display_name);
                }
            }
            ReplCommand::Sessions => print_sessions(&sessions_dir)?,
            ReplCommand::ResumeShow => {
                print_sessions(&sessions_dir)?;
                println!("use /resume <session_id> to resume a session");
            }
            ReplCommand::Resume(session_id) => {
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::ResumeSession(session_id),
                    CommandWait::UntilSessionResumed,
                    OutputMode::Streaming,
                )
                .await?;
            }
            ReplCommand::ReasoningShow => {
                let choices = model_metadata(config, &view.model_id)
                    .as_ref()
                    .map(reasoning_effort_choices)
                    .unwrap_or_else(|| "off".into());
                println!(
                    "reasoning effort: {}\navailable values: {}",
                    reasoning_effort_status_label(view.reasoning_effort.clone()),
                    choices
                );
            }
            ReplCommand::ReasoningSet(effort) => {
                let outcome = submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::SetReasoningEffort(effort.clone()),
                    CommandWait::UntilReasoningEffortChanged,
                    OutputMode::Streaming,
                )
                .await?;
                if outcome.error.is_some() || !outcome.notices.is_empty() {
                    continue;
                }
                println!(
                    "reasoning effort set to {}",
                    reasoning_effort_status_label(Some(effort))
                );
            }
            ReplCommand::Compact => {
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::Compact,
                    CommandWait::UntilDone,
                    OutputMode::Streaming,
                )
                .await?;
            }
            ReplCommand::ShowHistoryTree => {
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::ShowHistoryTree,
                    CommandWait::UntilHistoryLoaded,
                    OutputMode::Streaming,
                )
                .await?;
            }
            ReplCommand::Invalid(message) => {
                println!("{message}");
            }
            ReplCommand::NewSession => {
                submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::NewSession,
                    CommandWait::UntilSessionStarted,
                    OutputMode::Streaming,
                )
                .await?;
            }
            ReplCommand::Unsupported(message) => {
                println!("{message}");
            }
            ReplCommand::Prompt(input) => {
                let submission = UserMessageSubmission::new(
                    next_submission_id(),
                    UserMessageContent::new(input, Vec::new()),
                );
                let outcome = submit_and_wait(
                    &ingress,
                    &mut events,
                    &mut view,
                    SessionCommand::SubmitPrompt(submission),
                    CommandWait::UntilDone,
                    OutputMode::Streaming,
                )
                .await?;
                if let Some(error) = outcome.error {
                    return Err(anyhow!(error));
                }
                println!("\n");
            }
        }
    }

    let _ = ingress.shutdown();
    engine.join().await?;
    Ok(())
}

pub async fn run_one_shot(
    mut engine: SessionEngine,
    projection: SessionEngineProjection,
    prompt: String,
    json_output: bool,
) -> Result<()> {
    let ingress = engine.take_ingress();
    let mut events = engine.take_event_egress().into_receiver();
    let mut view = CliView::from_projection(projection, None);
    let started_at = Instant::now();

    let submission = UserMessageSubmission::new(
        next_submission_id(),
        UserMessageContent::new(prompt, Vec::new()),
    );
    let outcome = submit_and_wait(
        &ingress,
        &mut events,
        &mut view,
        SessionCommand::SubmitPrompt(submission),
        CommandWait::UntilDone,
        OutputMode::FinalOnly,
    )
    .await;
    let duration_ms = started_at.elapsed().as_millis();

    let _ = ingress.shutdown();
    let join_result = engine.join().await;

    match outcome {
        Ok(result) => {
            if let Some(error) = result.error {
                print_one_shot_error(&view, json_output, duration_ms, &error)?;
                let _ = join_result;
                return Err(anyhow!(error));
            }
            if json_output {
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "model": view.model_id,
                        "session_id": view.session_id,
                        "response": result.assistant,
                        "duration_ms": duration_ms,
                    })
                );
            } else {
                print!("{}", result.assistant);
                io::stdout().flush()?;
            }
            join_result?;
            Ok(())
        }
        Err(err) => {
            print_one_shot_error(&view, json_output, duration_ms, &format!("{err:#}"))?;
            let _ = join_result;
            Err(err)
        }
    }
}

struct WaitOutcome {
    assistant: String,
    error: Option<String>,
    notices: Vec<String>,
}

async fn submit_and_wait(
    ingress: &SessionEngineIngress,
    events: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
    view: &mut CliView,
    command: SessionCommand,
    wait: CommandWait,
    output_mode: OutputMode,
) -> Result<WaitOutcome> {
    ingress
        .submit(command)
        .map_err(|_| anyhow!("session engine command ingress closed"))?;
    wait_for_command(events, view, wait, output_mode).await
}

async fn wait_for_command(
    events: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
    view: &mut CliView,
    wait: CommandWait,
    output_mode: OutputMode,
) -> Result<WaitOutcome> {
    let interactive = matches!(output_mode, OutputMode::Streaming);
    let mut spinner = None;
    let mut compaction_pending = false;
    let mut assistant = String::new();
    let mut error = None;
    let mut notices = Vec::new();

    loop {
        let event = match events.recv().await {
            Some(event) => event,
            None => bail!("session engine event stream closed"),
        };

        let marker = present_transport_event(
            event,
            output_mode,
            interactive,
            &mut spinner,
            &mut compaction_pending,
            view,
            &mut assistant,
            &mut error,
            &mut notices,
        )?;

        let done = match wait {
            CommandWait::UntilDone => marker.done,
            CommandWait::UntilModelChanged => marker.model_changed || marker.error,
            CommandWait::UntilFastMode => marker.fast_mode || marker.notice || marker.error,
            CommandWait::UntilHistoryLoaded => marker.history_loaded || marker.error,
            CommandWait::UntilSessionResumed => {
                marker.session_resumed || marker.error || marker.notice
            }
            CommandWait::UntilSessionStarted => {
                marker.session_started || marker.error || marker.notice
            }
            CommandWait::UntilPermissionModeChanged => {
                marker.permission_mode_changed || marker.error
            }
            CommandWait::UntilReasoningEffortChanged => {
                marker.reasoning_effort_changed || marker.error || marker.notice
            }
        };
        if done {
            // Model switch may emit a follow-up missing-key Error after ModelChanged.
            if matches!(wait, CommandWait::UntilModelChanged) && marker.model_changed {
                drain_brief(
                    events,
                    output_mode,
                    interactive,
                    view,
                    &mut assistant,
                    &mut error,
                    &mut notices,
                )
                .await?;
            }
            break;
        }
    }

    if let Some(spinner) = spinner.take() {
        let _ = spinner.stop();
    }
    Ok(WaitOutcome {
        assistant,
        error,
        notices,
    })
}

async fn drain_brief(
    events: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
    output_mode: OutputMode,
    interactive: bool,
    view: &mut CliView,
    assistant: &mut String,
    error: &mut Option<String>,
    notices: &mut Vec<String>,
) -> Result<()> {
    let mut spinner = None;
    let mut compaction_pending = false;
    let deadline = Instant::now() + Duration::from_millis(80);
    while Instant::now() < deadline {
        match tokio::time::timeout(
            deadline.saturating_duration_since(Instant::now()),
            events.recv(),
        )
        .await
        {
            Ok(Some(event)) => {
                let _ = present_transport_event(
                    event,
                    output_mode,
                    interactive,
                    &mut spinner,
                    &mut compaction_pending,
                    view,
                    assistant,
                    error,
                    notices,
                )?;
            }
            Ok(None) | Err(_) => break,
        }
    }
    if let Some(spinner) = spinner.take() {
        let _ = spinner.stop();
    }
    Ok(())
}

#[derive(Default)]
struct EventMarker {
    done: bool,
    error: bool,
    notice: bool,
    model_changed: bool,
    fast_mode: bool,
    history_loaded: bool,
    session_resumed: bool,
    session_started: bool,
    permission_mode_changed: bool,
    reasoning_effort_changed: bool,
}

fn present_transport_event(
    event: SessionTransportEvent,
    output_mode: OutputMode,
    interactive: bool,
    spinner: &mut Option<ToolSpinner>,
    compaction_pending: &mut bool,
    view: &mut CliView,
    assistant: &mut String,
    last_error: &mut Option<String>,
    notices: &mut Vec<String>,
) -> Result<EventMarker> {
    let mut marker = EventMarker::default();
    match event {
        SessionTransportEvent::AssistantDelta(delta) => {
            assistant.push_str(&delta.delta);
            if matches!(output_mode, OutputMode::Streaming) {
                print!("{}", delta.delta);
                io::stdout().flush()?;
            }
        }
        SessionTransportEvent::ToolStarted(tool) => {
            if matches!(output_mode, OutputMode::Streaming) {
                *spinner = Some(ToolSpinner::start(tool.summary)?);
            }
        }
        SessionTransportEvent::ToolOutputDelta(delta) => {
            if matches!(output_mode, OutputMode::Streaming) {
                if let Some(active) = spinner.take() {
                    active.stop()?;
                }
                print!("{}", delta.chunk);
                io::stdout().flush()?;
            }
        }
        SessionTransportEvent::ToolFinished(tool) => {
            let ok = matches!(tool.outcome, ToolOutcome::Success);
            if let Some(active) = spinner.take() {
                active.finish(ok)?;
            } else if matches!(output_mode, OutputMode::Streaming) {
                let status = if ok { "✓" } else { "✗" };
                println!("-> {} {}", tool.name, status);
            }
        }
        SessionTransportEvent::CompactionStarted => {
            if matches!(output_mode, OutputMode::Streaming)
                && let Some(message) =
                    cli_compaction_lifecycle_message(compaction_pending, CompactionSignal::Started)
            {
                println!("{message}");
            }
        }
        SessionTransportEvent::CompactionNoProgress { blockers } => {
            if matches!(output_mode, OutputMode::Streaming)
                && let Some(message) = cli_compaction_lifecycle_message(
                    compaction_pending,
                    CompactionSignal::NoProgress { blockers },
                )
            {
                println!("{message}");
            }
        }
        SessionTransportEvent::CompactionFailed => {
            let _ = cli_compaction_lifecycle_message(compaction_pending, CompactionSignal::Failed);
        }
        SessionTransportEvent::CompactionCommitted { .. } => {
            if matches!(output_mode, OutputMode::Streaming)
                && let Some(message) = cli_compaction_lifecycle_message(
                    compaction_pending,
                    CompactionSignal::Committed,
                )
            {
                println!("{message}");
            }
        }
        SessionTransportEvent::Notice(notice) => {
            marker.notice = true;
            notices.push(notice.message.clone());
            if matches!(output_mode, OutputMode::Streaming) {
                println!("{}", notice.message);
            }
        }
        SessionTransportEvent::ProcessIssue(issue) => {
            if matches!(output_mode, OutputMode::Streaming) {
                println!("{}", issue.message);
            }
        }
        SessionTransportEvent::Error(err) => {
            marker.error = true;
            *last_error = Some(err.message.clone());
            if matches!(output_mode, OutputMode::Streaming) {
                println!("{}", err.message);
            }
        }
        SessionTransportEvent::Done => {
            marker.done = true;
        }
        SessionTransportEvent::FastModeChanged { .. } => {
            marker.fast_mode = true;
        }
        SessionTransportEvent::ModelChanged { model_id } => {
            marker.model_changed = true;
            view.model_id = model_id;
        }
        SessionTransportEvent::SettingChangeFailed { .. } => {}
        SessionTransportEvent::PermissionModeChanged { mode } => {
            marker.permission_mode_changed = true;
            view.permission_mode_label = mode.clone();
        }
        SessionTransportEvent::ReasoningEffortChanged { effort } => {
            marker.reasoning_effort_changed = true;
            view.reasoning_effort = Some(effort);
        }
        SessionTransportEvent::SessionHistoryLoaded { entries } => {
            marker.history_loaded = true;
            if matches!(output_mode, OutputMode::Streaming) {
                for entry in entries {
                    println!("{} {}", entry.id, entry.label);
                }
            }
        }
        SessionTransportEvent::SessionResumed {
            session_id,
            messages,
            evidence_count,
            model_id,
            ..
        } => {
            marker.session_resumed = true;
            view.session_id = session_id.clone();
            if let Some(model_id) = model_id {
                view.model_id = model_id.clone();
            }
            if matches!(output_mode, OutputMode::Streaming) {
                match &view.model_id {
                    model if !model.is_empty() => println!(
                        "resumed session {} ({} messages, {} evidence, model {})",
                        session_id,
                        messages.len(),
                        evidence_count,
                        model
                    ),
                    _ => println!(
                        "resumed session {} ({} messages, {} evidence)",
                        session_id,
                        messages.len(),
                        evidence_count
                    ),
                }
            }
        }
        SessionTransportEvent::SessionStarted { session_id, .. } => {
            marker.session_started = true;
            view.session_id = session_id.clone();
            if matches!(output_mode, OutputMode::Streaming) {
                println!("started new session {session_id}");
            }
        }
        SessionTransportEvent::PermissionRequested { event, handle } => {
            if interactive {
                respond_permission(&event, &handle)?;
            }
            // Non-interactive: drop handle → runner fails with sender dropped.
        }
        SessionTransportEvent::ChildPermissionRequested { event, handle, .. } => {
            if interactive {
                respond_permission(&event, &handle)?;
            }
        }
        SessionTransportEvent::QuestionRequested { request, handle } => {
            if interactive {
                match ask_questions_in_terminal(&request) {
                    Ok(response) => handle.answer(response)?,
                    Err(error) => {
                        let _ = handle.cancel(format!("{error:#}"));
                        return Err(error);
                    }
                }
            }
        }
        SessionTransportEvent::ChildQuestionRequested {
            request, handle, ..
        } => {
            if interactive {
                match ask_questions_in_terminal(&request) {
                    Ok(response) => handle.answer(response)?,
                    Err(error) => {
                        let _ = handle.cancel(format!("{error:#}"));
                        return Err(error);
                    }
                }
            }
        }
        _ => {}
    }
    Ok(marker)
}

fn respond_permission(
    event: &PermissionRequestEvent,
    handle: &crate::session::RunnerPermissionRequest,
) -> Result<()> {
    let approval = confirm_permission_event(event)?;
    match approval {
        PermissionApproval::AllowOnce => handle.approve(),
        PermissionApproval::AllowAlways => handle.allow_always(),
        PermissionApproval::Deny => handle.deny(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompactionSignal {
    Started,
    NoProgress { blockers: Vec<String> },
    Failed,
    Committed,
}

fn cli_compaction_lifecycle_message(
    pending: &mut bool,
    signal: CompactionSignal,
) -> Option<String> {
    match signal {
        CompactionSignal::Started => {
            *pending = true;
            Some("Compacting earlier messages…".into())
        }
        CompactionSignal::NoProgress { blockers } => {
            *pending = false;
            Some(format!(
                "Compaction made no progress: {}",
                blockers.join(",")
            ))
        }
        CompactionSignal::Failed => {
            *pending = false;
            None
        }
        CompactionSignal::Committed => {
            let committed = *pending;
            *pending = false;
            committed.then(|| "Earlier messages compacted".into())
        }
    }
}

#[cfg(test)]
pub(crate) fn record_one_shot_error(
    recorder: &std::sync::Arc<std::sync::Mutex<crate::transcript::TranscriptRecorder>>,
    err: &anyhow::Error,
) -> Result<()> {
    recorder
        .lock()
        .expect("transcript recorder poisoned")
        .record_error(format!("{err:#}"))
}

fn print_one_shot_error(
    view: &CliView,
    json_output: bool,
    duration_ms: u128,
    err: &str,
) -> Result<()> {
    if json_output {
        println!(
            "{}",
            json!({
                "ok": false,
                "model": view.model_id,
                "session_id": view.session_id,
                "response": "",
                "duration_ms": duration_ms,
                "error": err,
            })
        );
    }
    Ok(())
}

fn next_submission_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("cli-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReplCommand {
    Empty,
    Exit,
    Help,
    PermissionShow,
    PermissionSet(PermissionMode),
    ModelShow,
    ModelSet(String),
    ToggleFastMode,
    Sessions,
    ResumeShow,
    Resume(String),
    ReasoningShow,
    ReasoningSet(ModelReasoningEffort),
    Compact,
    ShowHistoryTree,
    Invalid(String),
    NewSession,
    Unsupported(String),
    Prompt(String),
}

fn parse_repl_command(input: &str) -> ReplCommand {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return ReplCommand::Empty;
    }

    if trimmed == "/sessions" {
        return ReplCommand::Sessions;
    }

    let intent = match parse_command(trimmed) {
        Ok(intent) => intent,
        Err(error) => return ReplCommand::Invalid(error.message().to_string()),
    };

    if let Some(session_command) = SessionCommand::from_command_intent(intent.clone()) {
        return repl_command_from_session_command(session_command);
    }

    match intent {
        CommandIntent::Help => ReplCommand::Help,
        CommandIntent::Exit => ReplCommand::Exit,
        CommandIntent::PermissionShow => ReplCommand::PermissionShow,
        CommandIntent::ModelShow => ReplCommand::ModelShow,
        CommandIntent::AgentsShow => ReplCommand::Unsupported(
            "CLI does not support /agents yet; use the TUI to select expert models.".into(),
        ),
        CommandIntent::ReasoningShow => ReplCommand::ReasoningShow,
        CommandIntent::ResumeShow => ReplCommand::ResumeShow,
        CommandIntent::ToolOutputSet(ToolOutputMode::Toggle)
        | CommandIntent::ToolOutputSet(ToolOutputMode::Expanded)
        | CommandIntent::ToolOutputSet(ToolOutputMode::Truncated) => ReplCommand::Unsupported(
            "CLI does not support /tool-output yet; parity is pending.".into(),
        ),
        CommandIntent::TranscriptScrollbarSet(_) => ReplCommand::Unsupported(
            "CLI does not support /scrollbar; use the TUI to toggle the transcript scrollbar."
                .into(),
        ),
        CommandIntent::Theme(_) => ReplCommand::Unsupported(
            "CLI does not support /theme; use the TUI to switch themes.".into(),
        ),
        CommandIntent::ContextBrowse => ReplCommand::Unsupported(
            "CLI does not support /context yet; use the TUI for context browsing.".into(),
        ),
        CommandIntent::McpBrowse | CommandIntent::SkillBrowse => {
            ReplCommand::Unsupported("CLI does not support this panel; use the TUI.".into())
        }
        CommandIntent::Prompt(_)
        | CommandIntent::Delegate { .. }
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

fn repl_command_from_session_command(command: SessionCommand) -> ReplCommand {
    match command {
        SessionCommand::SubmitPrompt(submission) => {
            let prompt = submission.text().to_string();
            if prompt.is_empty() {
                ReplCommand::Empty
            } else {
                ReplCommand::Prompt(prompt)
            }
        }
        SessionCommand::SetPermissionMode(mode) => ReplCommand::PermissionSet(mode),
        SessionCommand::SetModel(model_id) => ReplCommand::ModelSet(model_id),
        SessionCommand::SetExpertModel { .. } => ReplCommand::Unsupported(
            "CLI does not support expert model selection yet; use the TUI.".into(),
        ),
        SessionCommand::ToggleFastMode => ReplCommand::ToggleFastMode,
        SessionCommand::SetReasoningEffort(effort) => ReplCommand::ReasoningSet(effort),
        SessionCommand::Compact => ReplCommand::Compact,
        SessionCommand::ResumeSession(session_id) => ReplCommand::Resume(session_id),
        SessionCommand::NewSession => ReplCommand::NewSession,
        SessionCommand::ShowHistoryTree => ReplCommand::ShowHistoryTree,
        SessionCommand::Undo | SessionCommand::Redo | SessionCommand::NavigateHistory { .. } => {
            ReplCommand::Unsupported(
                "CLI does not support history navigation yet; use the TUI.".into(),
            )
        }
        SessionCommand::DelegateSubagent { .. } => ReplCommand::Unsupported(
            "CLI does not support @expert delegation yet; use the TUI for subagents.".into(),
        ),
        SessionCommand::ViewChild { .. } => ReplCommand::Unsupported(
            "CLI does not support /child or /children yet; child transcript parity is pending. Use the TUI for child navigation.".into(),
        ),
        SessionCommand::ViewParent => ReplCommand::Unsupported(
            "CLI does not support /parent yet; child transcript parity is pending. Use the TUI for child navigation.".into(),
        ),
        SessionCommand::ToggleMcpServer(_) | SessionCommand::Interrupt => ReplCommand::Unsupported(
            "CLI does not support this session command yet; use the TUI.".into(),
        ),
    }
}

fn reasoning_effort_label(effort: &ModelReasoningEffort) -> &str {
    effort.as_str()
}

fn reasoning_effort_choices(metadata: &ModelRequestMetadata) -> String {
    let efforts = metadata.selectable_reasoning_efforts();
    if efforts.is_empty() {
        return "off".into();
    }

    std::iter::once("off".to_string())
        .chain(
            efforts
                .into_iter()
                .filter(|effort| *effort != ModelReasoningEffort::None)
                .map(|effort| reasoning_effort_label(&effort).to_string()),
        )
        .collect::<Vec<_>>()
        .join(", ")
}

fn model_metadata(config: &AppConfig, model_display_name: &str) -> Option<ModelRequestMetadata> {
    let (provider_name, model_id) = model_display_name.split_once('/')?;
    config
        .providers
        .get(provider_name)
        .and_then(|provider| provider.models.get(model_id))
        .map(|model| model.request_metadata())
}

pub(crate) fn reasoning_effort_status_label(effort: Option<ModelReasoningEffort>) -> String {
    match effort {
        Some(ModelReasoningEffort::None) | None => "off".into(),
        Some(effort) => reasoning_effort_label(&effort).into(),
    }
}

pub(crate) fn parse_model_route(config: &AppConfig, input: &str) -> crate::config::ModelRoute {
    let active_route = config.active_route();
    if config.active_provider().1.has_model(input) {
        return crate::config::ModelRoute::new(active_route.provider, input);
    }
    match input.split_once('/') {
        Some((provider, model)) => crate::config::ModelRoute::new(provider, model),
        None => crate::config::ModelRoute::new(active_route.provider, input),
    }
}

fn print_repl_help() {
    println!("available commands:");
    for command in command_metadata()
        .iter()
        .filter(|command| command.visible_in_help)
    {
        println!("  {:<24} {}", command.usage, command.description);
    }
    println!("  /sessions                list sessions");
}

fn print_sessions(base_dir: &Path) -> Result<()> {
    let sessions = list_sessions(base_dir)?;

    if sessions.is_empty() {
        println!("no sessions");
        return Ok(());
    }

    for session in sessions {
        let model = session.model.unwrap_or_else(|| "unknown".to_string());
        let first = session
            .first_timestamp_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let last = session
            .last_timestamp_ms
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{}  {} records  {} -> {}  {}",
            session.session_id, session.record_count, first, last, model
        );

        if let Some(title) = session.title {
            println!("  title: {}", title);
        } else if let Some(summary) = session.last_user_summary {
            println!("  user: {}", summary);
        } else if let Some(summary) = session.last_assistant_summary {
            println!("  assistant: {}", summary);
        }
    }

    Ok(())
}

fn confirm_permission_event(event: &PermissionRequestEvent) -> Result<PermissionApproval> {
    println!();
    let class = event
        .rationale
        .as_deref()
        .and_then(|rationale| rationale.split_whitespace().next())
        .unwrap_or("tool");
    let detail = event.arguments.as_deref().unwrap_or(event.summary.as_str());
    println!(
        "permission required [{}]: {} {}",
        class, event.tool_name, detail
    );
    println!("summary: {}", event.summary);

    if event.can_allow_always {
        if let Some(summary) = &event.grant_summary {
            println!("session scope: {summary}");
        }
        print!("allow? [y=once/a=always/N] ");
    } else {
        print!("allow? [y=once/N] ");
    }
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(permission_approval_from_input(
        input.trim(),
        event.can_allow_always,
    ))
}

fn permission_approval_from_input(input: &str, can_allow_always: bool) -> PermissionApproval {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "o" | "yes" | "once" => PermissionApproval::AllowOnce,
        "a" | "always" if can_allow_always => PermissionApproval::AllowAlways,
        _ => PermissionApproval::Deny,
    }
}

fn ask_questions_in_terminal(request: &QuestionRequest) -> Result<QuestionResponse> {
    println!();
    println!("question tool requires your reply:");

    let mut answers = Vec::with_capacity(request.questions.len());
    for question in &request.questions {
        let selected = loop {
            println!();
            println!("{}", question.header);
            println!("{}", question.question);
            for (index, option) in question.options.iter().enumerate() {
                println!("  {}. {} — {}", index + 1, option.label, option.description);
            }
            println!("  0. Type your own answer");

            if question.multiple {
                print!("Select one or more options (comma separated), or 0 for custom: ");
            } else {
                print!("Select an option number, or 0 for custom: ");
            }
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input == "0" {
                print!("Your answer: ");
                io::stdout().flush()?;
                let mut custom = String::new();
                io::stdin().read_line(&mut custom)?;
                let custom = custom.trim();
                if custom.is_empty() {
                    println!("Please enter a non-empty answer.");
                    continue;
                }
                break vec![custom.to_string()];
            }

            if input.is_empty() {
                println!("Please answer the question before continuing.");
                continue;
            }

            if question.multiple {
                let selected: Vec<String> = input
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .filter_map(|part| part.parse::<usize>().ok())
                    .filter_map(|index| question.options.get(index.saturating_sub(1)))
                    .map(|option| option.label.clone())
                    .collect();
                if selected.is_empty() {
                    println!("Please choose at least one valid option.");
                    continue;
                }
                break selected;
            }

            let Some(option) = input
                .parse::<usize>()
                .ok()
                .and_then(|index| question.options.get(index.saturating_sub(1)))
            else {
                println!("Please choose a valid option number.");
                continue;
            };
            break vec![option.label.clone()];
        };

        answers.push(selected);
    }

    Ok(QuestionResponse { answers })
}

struct ToolSpinner {
    label: String,
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl ToolSpinner {
    fn start(label: String) -> Result<Self> {
        println!();

        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread_label = label.clone();

        let handle = thread::spawn(move || {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut index = 0;

            while !thread_stop.load(Ordering::Relaxed) {
                print!(
                    "\r\x1b[2K-> {} {}",
                    thread_label,
                    frames[index % frames.len()]
                );
                let _ = io::stdout().flush();
                index += 1;
                thread::sleep(Duration::from_millis(90));
            }
        });

        Ok(Self {
            label,
            stop,
            handle,
        })
    }

    fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
        print!("\r\x1b[2K");
        io::stdout().flush()?;
        Ok(())
    }

    fn finish(self, ok: bool) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();

        let status = if ok { "✓" } else { "✗" };
        println!("\r\x1b[2K-> {} {}", self.label, status);
        io::stdout().flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::PermissionApproval;

    fn cli_view() -> CliView {
        CliView {
            session_id: "session".into(),
            model_id: "provider/model".into(),
            model_label: "Model".into(),
            permission_mode_label: "default".into(),
            reasoning_effort: None,
        }
    }

    fn project_cli_event(view: &mut CliView, event: SessionTransportEvent) -> EventMarker {
        present_transport_event(
            event,
            OutputMode::FinalOnly,
            false,
            &mut None,
            &mut false,
            view,
            &mut String::new(),
            &mut None,
            &mut Vec::new(),
        )
        .expect("event projection")
    }

    #[test]
    fn permission_mode_changes_only_on_authoritative_event() {
        let mut view = cli_view();
        let marker = project_cli_event(
            &mut view,
            SessionTransportEvent::PermissionModeChanged {
                mode: "safe".into(),
            },
        );

        assert!(marker.permission_mode_changed);
        assert_eq!(view.permission_mode_label, "safe");
    }

    #[test]
    fn permission_error_does_not_change_cli_state() {
        let mut view = cli_view();
        let marker = project_cli_event(
            &mut view,
            SessionTransportEvent::Error(crate::session::ErrorEvent::new("failed")),
        );

        assert!(marker.error);
        assert_eq!(view.permission_mode_label, "default");
    }

    #[test]
    fn reasoning_effort_changes_only_on_authoritative_event() {
        let mut view = cli_view();
        let marker = project_cli_event(
            &mut view,
            SessionTransportEvent::ReasoningEffortChanged {
                effort: ModelReasoningEffort::High,
            },
        );

        assert!(marker.reasoning_effort_changed);
        assert_eq!(view.reasoning_effort, Some(ModelReasoningEffort::High));
    }

    #[test]
    fn terminal_permission_input_parses_once_always_and_denial() {
        assert_eq!(
            permission_approval_from_input("o", true),
            PermissionApproval::AllowOnce
        );
        assert_eq!(
            permission_approval_from_input("a", true),
            PermissionApproval::AllowAlways
        );
        assert_eq!(
            permission_approval_from_input("always", false),
            PermissionApproval::Deny
        );
        assert_eq!(
            permission_approval_from_input("n", true),
            PermissionApproval::Deny
        );
    }
}
