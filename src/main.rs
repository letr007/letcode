mod agent;
mod code_analysis;
mod config;
mod evidence;
mod mcp;
mod permission;
mod request_builder;
mod tool;
mod tool_format;
mod transcript;
mod tui;

use agent::{Agent, AgentEvent};
use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use config::AppConfig;
use permission::{PermissionMode, PermissionRequest};
use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::mpsc;
use tool_format::format_tool_call;
use tracing::warn;
use tracing_subscriber::EnvFilter;
use transcript::{
    TranscriptRecorder, list_sessions, read_records, remove_empty_session_file, resolve_session_id,
    restore_conversation_messages, restore_max_turn_id, restore_session_evidence,
};
use tui::runtime::AvailableModel;

#[tokio::main]
async fn main() -> Result<()> {
    let entry_mode = parse_entry_mode();
    let config = AppConfig::load()?;
    init_tracing(&config.global.log_file);

    let (active_provider_name, active_provider) = config.active_provider();
    let api_key_hint = format!(
        "Set providers.{active_provider_name}.api_key in {} or set {}.",
        config.config_path.display(),
        config.active_provider_api_key_env_var()
    );

    let api_base = active_provider.base_url.clone();
    let api_key = active_provider.api_key.clone();
    let api_key_configured = !api_key.trim().is_empty();
    let available_models = active_provider
        .models
        .iter()
        .map(|(model_id, model)| {
            AvailableModel::with_context_window_and_reasoning(
                model_id.clone(),
                active_provider.model_label(model_id),
                model.context_window,
                model.reasoning_effort,
            )
        })
        .collect::<Vec<_>>();
    let oai_config = OpenAIConfig::new()
        .with_api_base(api_base)
        .with_api_key(api_key);
    let client = Client::with_config(oai_config);
    let mut agent = Agent::new(
        client,
        active_provider.default_model.clone(),
        config.global.max_iterations,
        config.global.max_tool_calls,
    );
    agent.set_default_protocol(active_provider.protocol);
    let model_catalog = active_provider
        .models
        .iter()
        .map(|(model_id, model)| (model_id.clone(), model.request_metadata()))
        .collect::<HashMap<_, _>>();
    agent.set_model_catalog(model_catalog);
    let model_protocols = active_provider
        .models
        .iter()
        .map(|(model_id, model)| (model_id.clone(), model.protocol))
        .collect::<HashMap<_, _>>();
    agent.set_model_protocols(model_protocols);
    agent.set_permission_mode(config.permissions.mode);
    if matches!(config.permissions.mode, PermissionMode::Solo) {
        eprintln!(
            "warning: permissions.mode is set to 'solo'; write and command tools will run without confirmation"
        );
    }
    let recorder = Arc::new(Mutex::new(TranscriptRecorder::create(
        &config.global.sessions_dir,
    )?));

    {
        let mut recorder = recorder.lock().expect("transcript recorder poisoned");
        recorder.record_session_started(agent.model().to_string())?;
    }

    match entry_mode {
        EntryMode::Cli => {}
        EntryMode::Tui => {
            let (mcp_tools_tx, mcp_tools_rx) = mpsc::unbounded_channel();
            let mcp_config = config.mcp.clone();
            tokio::spawn(async move {
                let result = mcp::discover_tools(&mcp_config).await;
                let _ = mcp_tools_tx.send(result);
            });
            tui::run_tui(
                agent,
                recorder,
                config.global.sessions_dir.clone(),
                api_key_configured,
                api_key_hint,
                active_provider_name.to_string(),
                available_models,
                Some(mcp_tools_rx),
            )
            .await?;
            return Ok(());
        }
    }

    for tool in mcp::discover_tools(&config.mcp).await? {
        agent.try_register_tool(tool)?;
    }

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "exit" | "quit" => break,
            "" => continue,
            "/permission" | "/perm" => {
                println!(
                    "permission mode: {}\navailable modes: safe, default, solo",
                    agent.permission_mode()
                );
            }
            "/permission safe" | "/perm safe" => {
                let previous = agent.permission_mode();
                agent.set_permission_mode(PermissionMode::Safe);
                if previous != PermissionMode::Safe {
                    recorder
                        .lock()
                        .expect("transcript recorder poisoned")
                        .record_permission_mode_changed(previous.to_string(), "safe")?;
                }
                println!("permission mode set to safe");
            }
            "/permission default" | "/perm default" => {
                let previous = agent.permission_mode();
                agent.set_permission_mode(PermissionMode::Default);
                if previous != PermissionMode::Default {
                    recorder
                        .lock()
                        .expect("transcript recorder poisoned")
                        .record_permission_mode_changed(previous.to_string(), "default")?;
                }
                println!("permission mode set to default");
            }
            "/model" => {
                println!(
                    "current model: {} ({})",
                    config.active_provider_model_label(agent.model()),
                    agent.model()
                );
                println!("available models:");
                for (model_id, _) in &active_provider.models {
                    println!("  {} ({})", active_provider.model_label(model_id), model_id);
                }
            }
            "/sessions" => {
                print_sessions(&config.global.sessions_dir)?;
            }
            "/permission solo" | "/perm solo" => {
                print!(
                    "solo mode allows write and command tools without asking. Enable solo mode? [y/N] "
                );
                io::stdout().flush()?;

                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                let confirm = confirm.trim().to_ascii_lowercase();

                if matches!(confirm.as_str(), "y" | "yes") {
                    let previous = agent.permission_mode();
                    agent.set_permission_mode(PermissionMode::Solo);
                    if previous != PermissionMode::Solo {
                        recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_permission_mode_changed(previous.to_string(), "solo")?;
                    }
                    println!("permission mode set to solo");
                } else {
                    println!("solo mode not enabled");
                }
            }
            _ if input.starts_with("/session resume ") || input.starts_with("/resume ") => {
                let prefix = if let Some(session_id) = input.strip_prefix("/session resume ") {
                    session_id.trim()
                } else if let Some(session_id) = input.strip_prefix("/resume ") {
                    session_id.trim()
                } else {
                    ""
                };

                resume_session(&mut agent, &recorder, &config.global.sessions_dir, prefix)?;
            }
            _ if input.starts_with("/model ") => {
                let model_id = input.trim_start_matches("/model").trim();
                if !active_provider.has_model(model_id) {
                    println!("unknown model: {model_id}");
                    println!("available models:");
                    for (available_model_id, _) in &active_provider.models {
                        println!(
                            "  {} ({})",
                            active_provider.model_label(available_model_id),
                            available_model_id
                        );
                    }
                    continue;
                }

                let previous = agent.model().to_string();
                agent.set_model(model_id.to_string());
                if previous != model_id {
                    recorder
                        .lock()
                        .expect("transcript recorder poisoned")
                        .record_model_changed(previous, model_id)?;
                }
                println!(
                    "model set to {} ({})",
                    active_provider.model_label(model_id),
                    model_id
                );
            }
            _ => {
                if !api_key_configured {
                    println!(
                        "API key is not configured for active provider '{}'. {}",
                        active_provider_name, api_key_hint
                    );
                    continue;
                }

                {
                    let mut recorder = recorder.lock().expect("transcript recorder poisoned");
                    recorder.record_user_message(input.to_string())?;
                }

                let mut spinner: Option<ToolSpinner> = None;
                let event_recorder = Arc::clone(&recorder);
                let permission_recorder = Arc::clone(&recorder);

                let result = agent
                    .run_stream(
                        input,
                        |delta| {
                            print!("{}", delta);
                            io::stdout().flush()?;
                            Ok(())
                        },
                        |event| {
                            match event {
                                AgentEvent::TokenUsageUpdated { .. } => {}
                                AgentEvent::TurnStarted(event) => {
                                    if let Err(error) = event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_turn_started(event)
                                    {
                                        warn!(error = %error, "failed to record turn_started audit event");
                                    }
                                }
                                AgentEvent::EvidenceRecorded(evidence) => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_evidence_record(evidence)?;
                                }
                                AgentEvent::ReasoningDelta { .. } => {}
                                AgentEvent::ReasoningDone { text, .. } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_reasoning_message(text)?;
                                }
                                AgentEvent::ToolCallStarted {
                                    call_id,
                                    name,
                                    args,
                                } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_tool_call_started(
                                            call_id.clone(),
                                            name.clone(),
                                            args.clone(),
                                        )?;
                                    spinner =
                                        Some(ToolSpinner::start(format_tool_call(&name, &args))?);
                                }
                                AgentEvent::ToolCallFinished {
                                    call_id,
                                    name,
                                    ok,
                                    output,
                                } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_tool_call_finished(
                                            call_id.clone(),
                                            name.clone(),
                                            ok,
                                            output.clone(),
                                        )?;
                                    if let Some(spinner) = spinner.take() {
                                        spinner.finish(ok)?;
                                    } else {
                                        let status = if ok { "✓" } else { "✗" };
                                        println!("-> {} {}", name, status);
                                    }
                                }
                                AgentEvent::TodoSnapshotUpdated { items } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_todo_snapshot(items)?;
                                }
                                AgentEvent::AutoContinueChanged { state } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_auto_continue_changed(state)?;
                                }
                                AgentEvent::AutoContinuationScheduled {
                                    continuation_count,
                                    remaining_unfinished,
                                } => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_auto_continuation_scheduled(
                                            continuation_count,
                                            remaining_unfinished,
                                        )?;
                                }
                                AgentEvent::ValidationAdvisory(advisory) => {
                                    event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_validation_advisory(advisory)?;
                                }
                                AgentEvent::ToolExecutionSummary(event) => {
                                    if let Err(error) = event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_tool_execution_summary(event)
                                    {
                                        warn!(error = %error, "failed to record tool_execution_summary audit event");
                                    }
                                }
                                AgentEvent::TurnFinalized(event) => {
                                    if let Err(error) = event_recorder
                                        .lock()
                                        .expect("transcript recorder poisoned")
                                        .record_turn_finalized(event)
                                    {
                                        warn!(error = %error, "failed to record turn_finalized audit event");
                                    }
                                }
                            }

                            Ok(())
                        },
                        |request| {
                            let allowed = confirm_permission(&request)?;
                            permission_recorder
                                .lock()
                                .expect("transcript recorder poisoned")
                                .record_permission_decision(
                                    request.tool.clone(),
                                    request.args.clone(),
                                    allowed,
                                )?;
                            Ok(allowed)
                        },
                    )
                    .await;

                match result {
                    Ok(response) => {
                        recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_assistant_message(response)?;
                    }
                    Err(err) => {
                        recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_error(err.to_string())?;
                        return Err(err);
                    }
                }

                println!("\n");
            }
        }
    }

    remove_current_empty_session(&recorder)?;

    Ok(())
}

fn remove_current_empty_session(recorder: &Arc<Mutex<TranscriptRecorder>>) -> Result<bool> {
    let path = recorder
        .lock()
        .expect("transcript recorder poisoned")
        .path()
        .to_path_buf();

    remove_empty_session_file(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryMode {
    Cli,
    Tui,
}

fn parse_entry_mode() -> EntryMode {
    parse_entry_mode_from(env::args().skip(1))
}

fn parse_entry_mode_from<I, S>(args: I) -> EntryMode
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();

    match args.next().as_ref().map(|arg| arg.as_ref()) {
        Some("--cli") | Some("cli") | Some("repl") => EntryMode::Cli,
        Some("--tui") | Some("tui") | None => EntryMode::Tui,
        _ => EntryMode::Tui,
    }
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

        if let Some(summary) = session.last_user_summary {
            println!("  user: {}", summary);
        }

        if let Some(summary) = session.last_assistant_summary {
            println!("  assistant: {}", summary);
        }
    }

    Ok(())
}

fn resume_session<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: &Path,
    session_prefix: &str,
) -> Result<()> {
    if session_prefix.is_empty() {
        println!("usage: /resume <session_id>");
        return Ok(());
    }

    let sessions = list_sessions(sessions_dir)?;
    let session_id = match resolve_session_id(&sessions, session_prefix) {
        Ok(session_id) => session_id,
        Err(matches) if matches.is_empty() => {
            println!("session not found: {}", session_prefix);
            return Ok(());
        }
        Err(matches) => {
            println!("multiple sessions match {}:", session_prefix);
            for session_id in matches {
                println!("  {}", session_id);
            }
            return Ok(());
        }
    };

    let records = read_records(sessions_dir.join(format!("{session_id}.jsonl")))?;
    let messages = restore_conversation_messages(&records);
    let evidence = restore_session_evidence(&records)?;
    let max_turn_id = restore_max_turn_id(&records);
    let message_count = messages.len();
    let evidence_count = evidence.len();

    agent.restore_session_context(messages, evidence, max_turn_id)?;

    let new_recorder = TranscriptRecorder::open_existing(sessions_dir, &session_id)?;
    let new_path = new_recorder.path().to_path_buf();
    let old_path = recorder
        .lock()
        .expect("transcript recorder poisoned")
        .path()
        .to_path_buf();
    *recorder.lock().expect("transcript recorder poisoned") = new_recorder;

    if old_path != new_path {
        let _ = remove_empty_session_file(old_path);
    }

    println!(
        "resumed session {} ({} messages, {} evidence)",
        session_id, message_count, evidence_count
    );
    Ok(())
}

fn confirm_permission(request: &PermissionRequest) -> Result<bool> {
    println!();
    println!(
        "permission required [{}]: {}",
        request.class,
        format_tool_call(&request.tool, &request.args)
    );
    println!("summary: {}", request.summary);

    if let Some(preview) = &request.preview {
        println!("preview:\n{}", preview);
    }

    print!("allow? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_ascii_lowercase();

    Ok(matches!(input.as_str(), "y" | "yes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_mode_defaults_to_tui() {
        assert_eq!(
            parse_entry_mode_from(std::iter::empty::<&str>()),
            EntryMode::Tui
        );
    }

    #[test]
    fn entry_mode_supports_explicit_cli_and_tui() {
        assert_eq!(parse_entry_mode_from(["--cli"]), EntryMode::Cli);
        assert_eq!(parse_entry_mode_from(["cli"]), EntryMode::Cli);
        assert_eq!(parse_entry_mode_from(["repl"]), EntryMode::Cli);
        assert_eq!(parse_entry_mode_from(["--tui"]), EntryMode::Tui);
        assert_eq!(parse_entry_mode_from(["tui"]), EntryMode::Tui);
    }
}

fn init_tracing(log_path: &Path) {
    let log_file = match open_log_file(log_path) {
        Ok(log_file) => log_file,
        Err(err) => {
            eprintln!(
                "warning: failed to open log file {}: {}; tracing output will not be persisted",
                log_path.display(),
                err
            );
            return;
        }
    };

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("letcode=info,async_openai=warn"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(Mutex::new(log_file))
        .with_target(false)
        .with_ansi(false)
        .compact()
        .init();
}

fn open_log_file(log_path: &Path) -> io::Result<std::fs::File> {
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new().create(true).append(true).open(log_path)
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

    fn finish(self, ok: bool) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();

        let status = if ok { "✓" } else { "✗" };
        println!("\r\x1b[2K-> {} {}", self.label, status);
        io::stdout().flush()?;

        Ok(())
    }
}
