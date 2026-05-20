mod agent;
mod code_analysis;
mod permission;
mod tool;
mod tool_format;
mod transcript;
mod tui;

use agent::{Agent, AgentEvent};
use anyhow::Result;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use permission::{PermissionMode, PermissionRequest};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tool_format::format_tool_call;
use tracing_subscriber::EnvFilter;
use transcript::{
    TranscriptRecorder, list_sessions, read_records, resolve_session_id,
    restore_conversation_messages,
};

const SESSIONS_DIR: &str = "sessions";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let entry_mode = parse_entry_mode();

    let api_base =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.to_string());
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    let api_key_configured = !api_key.trim().is_empty();
    let oai_config = OpenAIConfig::new()
        .with_api_base(api_base)
        .with_api_key(api_key);
    let client = Client::with_config(oai_config);
    let mut agent = Agent::new(client, "gpt-5.5", 64, 128);
    let recorder = Arc::new(Mutex::new(TranscriptRecorder::create(SESSIONS_DIR)?));

    {
        let mut recorder = recorder.lock().expect("transcript recorder poisoned");
        recorder.record_session_started(agent.model().to_string())?;
    }

    match entry_mode {
        EntryMode::Cli => {}
        EntryMode::Tui => {
            tui::run_tui(agent, recorder, api_key_configured).await?;
            return Ok(());
        }
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
                agent.set_permission_mode(PermissionMode::Safe);
                println!("permission mode set to safe");
            }
            "/permission default" | "/perm default" => {
                agent.set_permission_mode(PermissionMode::Default);
                println!("permission mode set to default");
            }
            "/sessions" => {
                print_sessions()?;
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
                    agent.set_permission_mode(PermissionMode::Solo);
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

                resume_session(&mut agent, &recorder, prefix)?;
            }
            _ => {
                if !api_key_configured {
                    println!(
                        "OPENAI_API_KEY is not set. Set it and restart letcode before sending model requests."
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
                                            call_id,
                                            name.clone(),
                                            ok,
                                            output,
                                        )?;
                                    if let Some(spinner) = spinner.take() {
                                        spinner.finish(ok)?;
                                    } else {
                                        let status = if ok { "✓" } else { "✗" };
                                        println!("-> {} {}", name, status);
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

    Ok(())
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

fn print_sessions() -> Result<()> {
    let sessions = list_sessions(SESSIONS_DIR)?;

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

fn resume_session(
    agent: &mut Agent<OpenAIConfig>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    session_prefix: &str,
) -> Result<()> {
    if session_prefix.is_empty() {
        println!("usage: /resume <session_id>");
        return Ok(());
    }

    let sessions = list_sessions(SESSIONS_DIR)?;
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

    let records = read_records(format!("{SESSIONS_DIR}/{session_id}.jsonl"))?;
    let messages = restore_conversation_messages(&records);
    let message_count = messages.len();

    agent.restore_transcript_messages(messages);

    let new_recorder = TranscriptRecorder::open_existing(SESSIONS_DIR, &session_id)?;
    *recorder.lock().expect("transcript recorder poisoned") = new_recorder;

    println!(
        "resumed session {} ({} messages)",
        session_id, message_count
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

fn init_tracing() {
    let Ok(log_file) = open_log_file() else {
        return;
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

fn open_log_file() -> io::Result<std::fs::File> {
    fs::create_dir_all("logs")?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/combined.log")
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
