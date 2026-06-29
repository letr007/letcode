mod agent;
mod code_analysis;
mod command;
mod config;
mod context_view;
mod context_tree;
mod delegation;
mod evidence;
mod langfuse_trace;
mod mcp;
mod memory;
mod permission;
mod request_builder;
mod retry;
mod skills;
mod subagent;
mod subagent_events;
mod tool;
mod tool_format;
mod tool_names;
mod transcript;
mod tui;
mod user_content;

use agent::{Agent, AgentEvent, ManualCompactionOutcome};
use anyhow::{Result, anyhow, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use command::{CommandIntent, ToolOutputMode, command_metadata, parse_command};
use config::AppConfig;
use delegation::supported_agent_names;
use indexmap::IndexMap;
use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::runtime::Tokio;
use opentelemetry_sdk::trace::{
    SdkTracerProvider, span_processor_with_async_runtime::BatchSpanProcessor,
};
use permission::{PermissionMode, PermissionRequest};
use request_builder::ModelReasoningEffort;
use serde_json::json;
use skills::SkillRegistry;
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
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tool_format::format_tool_call;
use tracing::warn;
use tracing_subscriber::{
    EnvFilter, filter,
    layer::{Layer, SubscriberExt},
    util::SubscriberInitExt,
};
use transcript::{
    TranscriptRecorder, list_sessions, read_records, remove_empty_session_file, resolve_session_id,
    restore_job_board, transcript_has_session_title, transcript_has_user_message,
    transcript_projection,
};
use tui::runtime::AvailableModel;

#[tokio::main]
async fn main() -> Result<()> {
    let options = CliOptions::parse()?;
    dotenvy::dotenv().ok();
    let config = AppConfig::load()?;
    let _tracing_guards = init_tracing(&config.global.log_file);

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
    memory::set_memory_sessions_dir(config.global.sessions_dir.clone());
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
    agent.set_compaction_config(config.global.compaction.clone());
    agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
    agent.set_retry_config(
        active_provider
            .retry
            .clone()
            .unwrap_or_else(|| config.global.retry.clone()),
    );
    agent.set_permission_mode(config.permissions.mode);
    for agent_name in supported_agent_names() {
        if let Some(model) = config.agents.model_for(agent_name) {
            agent.set_subagent_model_override(agent_name, model.to_string());
        }
    }
    let skill_registry = Arc::new(SkillRegistry::load(
        &config.config_dir,
        &env::current_dir()?,
    )?);
    agent.register_skill_registry(skill_registry)?;
    if matches!(config.permissions.mode, PermissionMode::Solo) {
        eprintln!(
            "warning: permissions.mode is set to 'solo'; write and command tools will run without confirmation"
        );
    }
    let recorder = Arc::new(Mutex::new(TranscriptRecorder::create(
        &config.global.sessions_dir,
    )?));
    agent.set_context_scope_state(
        recorder
            .lock()
            .expect("transcript recorder poisoned")
            .context_scope_state(),
    );

    {
        let mut recorder = recorder.lock().expect("transcript recorder poisoned");
        recorder.record_session_started(agent.model().to_string())?;
        sync_agent_context_scope_from_recorder(&mut agent, &recorder)?;
    }

    match options.entry_mode {
        EntryMode::Cli { prompt, json } => {
            if let Some(prompt) = prompt {
                run_one_shot(
                    &mut agent,
                    &recorder,
                    &config.mcp,
                    &active_provider_name,
                    &api_key_hint,
                    api_key_configured,
                    prompt,
                    json,
                )
                .await?;
                remove_current_empty_session(&recorder)?;
                return Ok(());
            }
        }
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
                config.config_dir.clone(),
                api_key_configured,
                api_key_hint,
                active_provider_name.to_string(),
                available_models,
                langfuse_startup_toast(&_tracing_guards.langfuse_status),
                Some(mcp_tools_rx),
            )
            .await?;
            return Ok(());
        }
    }

    for tool in mcp::discover_tools(&config.mcp).await? {
        agent.try_register_tool(tool)?;
    }

    run_repl(
        &mut agent,
        &recorder,
        &config,
        &config.global.sessions_dir,
        active_provider_name,
        api_key_configured,
        &api_key_hint,
    )
    .await?;

    remove_current_empty_session(&recorder)?;

    Ok(())
}

async fn run_repl<C: async_openai::config::Config + Clone>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    config: &AppConfig,
    sessions_dir: &Path,
    active_provider_name: &str,
    api_key_configured: bool,
    api_key_hint: &str,
) -> Result<()> {
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
                    "permission mode: {}\navailable modes: safe, default, solo",
                    agent.permission_mode()
                );
            }
            ReplCommand::PermissionSet(PermissionMode::Safe) => {
                set_permission_mode(agent, recorder, PermissionMode::Safe)?;
                println!("permission mode set to safe");
            }
            ReplCommand::PermissionSet(PermissionMode::Default) => {
                set_permission_mode(agent, recorder, PermissionMode::Default)?;
                println!("permission mode set to default");
            }
            ReplCommand::PermissionSet(PermissionMode::Solo) => {
                print!(
                    "solo mode allows write and command tools without asking. Enable solo mode? [y/N] "
                );
                io::stdout().flush()?;

                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                let confirm = confirm.trim().to_ascii_lowercase();

                if matches!(confirm.as_str(), "y" | "yes") {
                    set_permission_mode(agent, recorder, PermissionMode::Solo)?;
                    println!("permission mode set to solo");
                } else {
                    println!("solo mode not enabled");
                }
            }
            ReplCommand::ModelShow => {
                let (_, active_provider) = config.active_provider();
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
            ReplCommand::ModelSet(model_id) => {
                let (_, active_provider) = config.active_provider();
                if !active_provider.has_model(&model_id) {
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
                agent.set_model(model_id.clone());
                if previous != model_id {
                    recorder
                        .lock()
                        .expect("transcript recorder poisoned")
                        .record_model_changed(previous, &model_id)?;
                }
                println!(
                    "model set to {} ({})",
                    active_provider.model_label(&model_id),
                    model_id
                );
            }
            ReplCommand::Sessions => print_sessions(sessions_dir)?,
            ReplCommand::ResumeShow => {
                print_sessions(sessions_dir)?;
                println!("use /resume <session_id> to resume a session");
            }
            ReplCommand::Resume(session_id) => {
                resume_session(agent, recorder, sessions_dir, &session_id)?;
            }
            ReplCommand::ReasoningShow => {
                println!(
                    "reasoning effort: {}\navailable values: off, none, minimal, low, medium, high, xhigh",
                    reasoning_effort_status_label(agent.reasoning_effort())
                );
            }
            ReplCommand::ReasoningSet(effort) => {
                agent.set_reasoning_effort(effort);
                println!(
                    "reasoning effort set to {}",
                    reasoning_effort_status_label(Some(effort))
                );
            }
            ReplCommand::Compact => {
                if !api_key_configured {
                    println!(
                        "API key is not configured for active provider '{}'. {}",
                        active_provider_name, api_key_hint
                    );
                    continue;
                }
                compact_agent_context(agent, recorder).await?;
            }
            ReplCommand::Invalid(message) => {
                println!("{message}");
            }
            ReplCommand::NewSession => {
                start_new_session(agent, recorder, sessions_dir)?;
                println!(
                    "started new session {}",
                    recorder
                        .lock()
                        .expect("transcript recorder poisoned")
                        .session_id()
                );
            }
            ReplCommand::Unsupported(message) => {
                println!("{message}");
            }
            ReplCommand::Prompt(input) => {
                if !api_key_configured {
                    println!(
                        "API key is not configured for active provider '{}'. {}",
                        active_provider_name, api_key_hint
                    );
                    continue;
                }

                let response =
                    run_agent_prompt(agent, recorder, &input, OutputMode::Streaming).await;

                match response {
                    Ok(_) => println!("\n"),
                    Err(err) => return Err(err),
                }
            }
        }
    }

    Ok(())
}

async fn run_one_shot<C: async_openai::config::Config + Clone>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    mcp_config: &IndexMap<String, config::McpServerConfig>,
    active_provider_name: &str,
    api_key_hint: &str,
    api_key_configured: bool,
    prompt: String,
    json_output: bool,
) -> Result<()> {
    if !api_key_configured {
        let err = anyhow!(
            "API key is not configured for active provider '{}'. {}",
            active_provider_name,
            api_key_hint
        );
        record_one_shot_error(recorder, &err)?;
        print_one_shot_error(recorder, agent.model(), json_output, 0, &err)?;
        return Err(err);
    }

    let tools = match mcp::discover_tools(mcp_config).await {
        Ok(tools) => tools,
        Err(err) => {
            record_one_shot_error(recorder, &err)?;
            print_one_shot_error(recorder, agent.model(), json_output, 0, &err)?;
            return Err(err);
        }
    };

    for tool in tools {
        if let Err(err) = agent.try_register_tool(tool) {
            record_one_shot_error(recorder, &err)?;
            print_one_shot_error(recorder, agent.model(), json_output, 0, &err)?;
            return Err(err);
        }
    }

    let started_at = Instant::now();
    let result = run_agent_prompt(agent, recorder, &prompt, OutputMode::FinalOnly).await;
    let duration_ms = started_at.elapsed().as_millis();

    match result {
        Ok(response) => {
            if json_output {
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "model": agent.model(),
                        "session_id": current_session_id(recorder),
                        "response": response,
                        "duration_ms": duration_ms,
                    })
                );
            } else {
                print!("{}", response);
                io::stdout().flush()?;
            }
            Ok(())
        }
        Err(err) => {
            print_one_shot_error(recorder, agent.model(), json_output, duration_ms, &err)?;
            Err(err)
        }
    }
}

async fn run_agent_prompt<C: async_openai::config::Config + Clone>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    input: &str,
    output_mode: OutputMode,
) -> Result<String> {
    let pending_title = pending_session_title(agent, recorder)?;
    {
        let mut recorder = recorder.lock().expect("transcript recorder poisoned");
        recorder.record_user_message(input.to_string())?;
    }
    if let Some((session_id, mut title_agent)) = pending_title {
        match title_agent.generate_session_title(input).await {
            Ok(title) => {
                let mut recorder = recorder.lock().expect("transcript recorder poisoned");
                if recorder.session_id() == session_id {
                    if let Err(error) = recorder.record_session_title(title) {
                        warn!(error = %error, session_id, "failed to persist generated session title");
                    }
                }
            }
            Err(error) => warn!(error = %error, session_id, "failed to generate session title"),
        }
    }

    let mut spinner: Option<ToolSpinner> = None;
    let event_recorder = Arc::clone(recorder);
    let permission_recorder = Arc::clone(recorder);
    let interactive_permissions = matches!(output_mode, OutputMode::Streaming);

    let result = agent
        .run_stream(
            input,
            |delta| {
                if matches!(output_mode, OutputMode::Streaming) {
                    print!("{}", delta);
                    io::stdout().flush()?;
                }
                Ok(())
            },
            |event| {
                match event {
                    AgentEvent::ContextCompactionStarted => {}
                    AgentEvent::ContextCompactionDelta { .. } => {}
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
                    AgentEvent::ModelStreamIssue { .. } => {}
                    AgentEvent::ReasoningDelta { .. } => {}
                    AgentEvent::ReasoningDone { text, .. } => {
                        event_recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_reasoning_message(text)?;
                    }
                    AgentEvent::ToolCallPending { .. } => {}
                    AgentEvent::ToolCallStarted { call_id, name, args } => {
                        event_recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_tool_call_started(call_id.clone(), name.clone(), args.clone())?;
                        if matches!(output_mode, OutputMode::Streaming) {
                            spinner = Some(ToolSpinner::start(format_tool_call(&name, &args))?);
                        }
                    }
                    AgentEvent::ToolCallCancelled { call_id, name } => {
                        event_recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_tool_call_cancelled(call_id, name)?;
                    }
                    AgentEvent::ToolOutputDelta { chunk, .. } => {
                        if matches!(output_mode, OutputMode::Streaming) {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop()?;
                            }
                            print!("{chunk}");
                            io::stdout().flush()?;
                        }
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
                            .record_tool_call_finished_and_apply_context_control(
                                call_id.clone(),
                                name.clone(),
                                ok,
                                output.clone(),
                            )?;
                        if let Some(spinner) = spinner.take() {
                            spinner.finish(ok)?;
                        } else if matches!(output_mode, OutputMode::Streaming) {
                            let status = if ok { "✓" } else { "✗" };
                            println!("-> {} {}", name, status);
                        }
                    }
                    AgentEvent::ToolCallBatchFinished => {}
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
                    AgentEvent::ContextCompacted(event) => {
                        event_recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_context_compaction(event)?;
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
                if interactive_permissions {
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
                } else {
                    bail!(
                        "permission required in non-interactive CLI mode [{}]: {}",
                        request.class,
                        format_tool_call(&request.tool, &request.args)
                    );
                }
            },
        )
        .await;

    match result {
        Ok(response) => {
            recorder
                .lock()
                .expect("transcript recorder poisoned")
                .record_assistant_message(response.clone())?;
            Ok(response)
        }
        Err(err) => {
            recorder
                .lock()
                .expect("transcript recorder poisoned")
                .record_error(err.to_string())?;
            Err(err)
        }
    }
}

async fn compact_agent_context<C: async_openai::config::Config + Clone>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<()> {
    let event_recorder = Arc::clone(recorder);
    let compacted_summary = Arc::new(Mutex::new(None::<String>));
    let printed_summary = AtomicBool::new(false);
    let streamed_summary = AtomicBool::new(false);
    match agent
        .compact_session_stream_async(
            |event| {
                let event_recorder = Arc::clone(&event_recorder);
                let compacted_summary = Arc::clone(&compacted_summary);
                async move {
                    if let AgentEvent::ContextCompacted(event) = event {
                        if let Ok(mut summary) = compacted_summary.lock() {
                            *summary = Some(event.summary.clone());
                        }
                        event_recorder
                            .lock()
                            .expect("transcript recorder poisoned")
                            .record_context_compaction(event)?;
                    }
                    Ok(())
                }
            },
            || {
                println!("──────── Context compacting ────────");
                printed_summary.store(true, Ordering::Release);
                Ok(())
            },
            |delta| {
                streamed_summary.store(true, Ordering::Release);
                print!("{delta}");
                io::stdout().flush().map_err(Into::into)
            },
        )
        .await?
    {
        ManualCompactionOutcome::Compacted { retained_items } => {
            if !streamed_summary.load(Ordering::Acquire) {
                if let Ok(summary) = compacted_summary.lock()
                    && let Some(summary) = summary.as_ref()
                {
                    println!("{summary}");
                }
            }
            if printed_summary.load(Ordering::Acquire) {
                println!();
                println!("──────── Context compacted ────────");
            }
            println!(
                "context compacted ({} history items retained)",
                retained_items
            );
        }
        ManualCompactionOutcome::NothingToCompact => {
            println!("nothing to compact yet");
        }
    }
    Ok(())
}

fn set_permission_mode<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    mode: PermissionMode,
) -> Result<()> {
    let previous = agent.permission_mode();
    agent.set_permission_mode(mode);
    if previous != mode {
        recorder
            .lock()
            .expect("transcript recorder poisoned")
            .record_permission_mode_changed(previous.to_string(), mode.to_string())?;
    }
    Ok(())
}

fn start_new_session<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    sessions_dir: &Path,
) -> Result<()> {
    let new_recorder = TranscriptRecorder::create(sessions_dir)?;
    let new_path = new_recorder.path().to_path_buf();
    if let Err(err) = agent.restore_session_context(Vec::new(), Vec::new(), 0) {
        let _ = remove_empty_session_file(new_path);
        return Err(err);
    }
    let old_path = recorder
        .lock()
        .expect("transcript recorder poisoned")
        .path()
        .to_path_buf();
    *recorder.lock().expect("transcript recorder poisoned") = new_recorder;
    recorder
        .lock()
        .expect("transcript recorder poisoned")
        .record_session_started(agent.model().to_string())?;
    sync_agent_context_scope_from_recorder(
        agent,
        &recorder.lock().expect("transcript recorder poisoned"),
    )?;
    let _ = remove_empty_session_file(old_path);
    Ok(())
}

fn current_session_id(recorder: &Arc<Mutex<TranscriptRecorder>>) -> String {
    recorder
        .lock()
        .expect("transcript recorder poisoned")
        .session_id()
        .to_string()
}

fn record_one_shot_error(
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    err: &anyhow::Error,
) -> Result<()> {
    recorder
        .lock()
        .expect("transcript recorder poisoned")
        .record_error(err.to_string())
}

fn print_one_shot_error(
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    model: &str,
    json_output: bool,
    duration_ms: u128,
    err: &anyhow::Error,
) -> Result<()> {
    if json_output {
        println!(
            "{}",
            json!({
                "ok": false,
                "model": model,
                "session_id": current_session_id(recorder),
                "response": "",
                "duration_ms": duration_ms,
                "error": err.to_string(),
            })
        );
        Ok(())
    } else {
        Ok(())
    }
}

fn remove_current_empty_session(recorder: &Arc<Mutex<TranscriptRecorder>>) -> Result<bool> {
    let path = recorder
        .lock()
        .expect("transcript recorder poisoned")
        .path()
        .to_path_buf();

    remove_empty_session_file(path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EntryMode {
    Cli { prompt: Option<String>, json: bool },
    Tui,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    entry_mode: EntryMode,
}

impl CliOptions {
    fn parse() -> Result<Self> {
        Self::parse_from(env::args().skip(1))
    }

    fn parse_from<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut entry_mode = EntryMode::Tui;
        let mut json = false;
        let mut prompt: Option<String> = None;
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "--cli" | "cli" | "repl" => {
                    entry_mode = EntryMode::Cli { prompt: None, json };
                }
                "--tui" | "tui" => {
                    if prompt.is_none() {
                        entry_mode = EntryMode::Tui;
                    }
                }
                "--json" => {
                    json = true;
                }
                "--prompt" | "-p" => {
                    let Some(value) = args.next() else {
                        bail!("{} requires a prompt value", arg.as_ref());
                    };
                    let value = value.as_ref();
                    if value.is_empty() || value.starts_with('-') {
                        bail!("{} requires a prompt value", arg.as_ref());
                    }
                    prompt = Some(value.to_string());
                }
                value if value.starts_with('-') => bail!("unknown flag: {value}"),
                value => bail!("unknown argument: {value}"),
            }
        }

        if prompt.is_some() {
            entry_mode = EntryMode::Cli {
                prompt: prompt.clone(),
                json,
            };
        } else if matches!(entry_mode, EntryMode::Cli { .. }) {
            entry_mode = EntryMode::Cli { prompt: None, json };
        }

        if json
            && !matches!(
                entry_mode,
                EntryMode::Cli {
                    prompt: Some(_),
                    ..
                }
            )
        {
            bail!("--json requires --prompt/-p in one-shot mode");
        }

        Ok(Self { entry_mode })
    }
}

fn parse_repl_command(input: &str) -> ReplCommand {
    let trimmed = input.trim();

    if trimmed.is_empty() {
        return ReplCommand::Empty;
    }

    if trimmed == "/sessions" {
        return ReplCommand::Sessions;
    }

    match parse_command(trimmed) {
        Ok(CommandIntent::Prompt(prompt)) if prompt.is_empty() => ReplCommand::Empty,
        Ok(CommandIntent::Prompt(prompt)) => ReplCommand::Prompt(prompt),
        Ok(CommandIntent::Help) => ReplCommand::Help,
        Ok(CommandIntent::Exit) => ReplCommand::Exit,
        Ok(CommandIntent::PermissionShow) => ReplCommand::PermissionShow,
        Ok(CommandIntent::PermissionSet(mode)) => ReplCommand::PermissionSet(mode),
        Ok(CommandIntent::ModelShow) => ReplCommand::ModelShow,
        Ok(CommandIntent::ModelSet(model_id)) => ReplCommand::ModelSet(model_id),
        Ok(CommandIntent::ReasoningShow) => ReplCommand::ReasoningShow,
        Ok(CommandIntent::ReasoningSet(effort)) => ReplCommand::ReasoningSet(effort),
        Ok(CommandIntent::Compact) => ReplCommand::Compact,
        Ok(CommandIntent::Tree) => ReplCommand::Unsupported(
            "CLI does not support /tree yet; use the TUI for context tree navigation.".into(),
        ),
        Ok(CommandIntent::Branches) => ReplCommand::Unsupported(
            "CLI does not support /branches yet; use the TUI for context branch commands."
                .into(),
        ),
        Ok(CommandIntent::BranchCreate(_)) => ReplCommand::Unsupported(
            "CLI does not support /branch yet; use the TUI for context branch commands.".into(),
        ),
        Ok(CommandIntent::CheckoutBranch(_)) => ReplCommand::Unsupported(
            "CLI does not support /checkout yet; use the TUI for context branch commands."
                .into(),
        ),
        Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Toggle))
        | Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Expanded))
        | Ok(CommandIntent::ToolOutputSet(ToolOutputMode::Truncated)) => ReplCommand::Unsupported(
            "CLI does not support /tool-output yet; parity is pending.".into(),
        ),
        Ok(CommandIntent::TranscriptScrollbarSet(_)) => ReplCommand::Unsupported(
            "CLI does not support /scrollbar; use the TUI to toggle the transcript scrollbar."
                .into(),
        ),
        Ok(CommandIntent::ResumeShow) => ReplCommand::ResumeShow,
        Ok(CommandIntent::Resume(session_id)) => ReplCommand::Resume(session_id),
        Ok(CommandIntent::NewSession) => ReplCommand::NewSession,
        Ok(CommandIntent::Delegate { .. }) => ReplCommand::Unsupported(
            "CLI does not support @expert delegation yet; use the TUI for subagents.".into(),
        ),
        Ok(CommandIntent::Child(_)) => ReplCommand::Unsupported(
            "CLI does not support /child or /children yet; child transcript parity is pending. Use the TUI for child navigation.".into(),
        ),
        Ok(CommandIntent::Parent) => ReplCommand::Unsupported(
            "CLI does not support /parent yet; child transcript parity is pending. Use the TUI for child navigation.".into(),
        ),
        Err(error) => ReplCommand::Invalid(error.message().to_string()),
    }
}

fn reasoning_effort_label(effort: ModelReasoningEffort) -> &'static str {
    match effort {
        ModelReasoningEffort::None => "none",
        ModelReasoningEffort::Minimal => "minimal",
        ModelReasoningEffort::Low => "low",
        ModelReasoningEffort::Medium => "medium",
        ModelReasoningEffort::High => "high",
        ModelReasoningEffort::Xhigh => "xhigh",
    }
}

fn reasoning_effort_status_label(effort: Option<ModelReasoningEffort>) -> &'static str {
    match effort {
        Some(ModelReasoningEffort::None) | None => "off",
        Some(effort) => reasoning_effort_label(effort),
    }
}

fn sync_agent_context_scope_from_recorder<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &TranscriptRecorder,
) -> Result<()> {
    agent.set_context_scope_state(recorder.context_scope_state());
    if let Some(scope) = recorder.active_context_experiment() {
        let snapshot = transcript_projection::build_session_context_snapshot(
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            None,
            transcript_projection::SessionContextCursor {
                branch_id: Some(scope.parent_branch_id.clone()),
                leaf_sequence: Some(scope.base_sequence),
            },
        )?;
        agent.set_context_experiment_restore_point(
            scope,
            snapshot.history,
            snapshot.evidence,
            snapshot.max_turn_id,
        );
    } else {
        agent.clear_context_experiment_restore_point();
    }
    Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReplCommand {
    Empty,
    Exit,
    Help,
    PermissionShow,
    PermissionSet(PermissionMode),
    ModelShow,
    ModelSet(String),
    Sessions,
    ResumeShow,
    Resume(String),
    ReasoningShow,
    ReasoningSet(ModelReasoningEffort),
    Compact,
    Invalid(String),
    NewSession,
    Unsupported(String),
    Prompt(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Streaming,
    FinalOnly,
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

fn pending_session_title<C: async_openai::config::Config + Clone>(
    agent: &Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
) -> Result<Option<(String, Agent<C>)>> {
    let (session_id, path) = {
        let recorder = recorder.lock().expect("transcript recorder poisoned");
        (
            recorder.session_id().to_string(),
            recorder.path().to_path_buf(),
        )
    };
    let records = read_records(&path)?;
    if transcript_has_user_message(&records) || transcript_has_session_title(&records) {
        return Ok(None);
    }

    Ok(Some((session_id, agent.session_title_agent())))
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
    let job_board = restore_job_board(sessions_dir, &records)?;
    let snapshot =
        transcript_projection::project_session_restore_snapshot(session_id.clone(), records, None)?;
    let message_count = snapshot.messages.len();
    let evidence_count = snapshot.evidence_count();

    if let Some(model) = &snapshot.latest_model {
        agent.set_model(model.clone());
    }
    agent.restore_session_history(snapshot.history, snapshot.evidence, snapshot.max_turn_id)?;

    let mut new_recorder = TranscriptRecorder::open_existing(sessions_dir, &session_id)?;
    if snapshot.branch_id == crate::transcript::ROOT_CONTEXT_BRANCH_ID {
        new_recorder.set_current_context_branch_id(None);
    } else {
        new_recorder.set_current_context_branch_id(Some(snapshot.branch_id.clone()));
    }
    sync_agent_context_scope_from_recorder(agent, &new_recorder)?;
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

    match snapshot.latest_model {
        Some(model) => println!(
            "resumed session {} ({} messages, {} evidence, model {})",
            session_id, message_count, evidence_count, model
        ),
        None => println!(
            "resumed session {} ({} messages, {} evidence)",
            session_id, message_count, evidence_count
        ),
    }
    if !job_board.is_empty() {
        let active = job_board.iter().filter(|job| job.active).count();
        let unreconciled = job_board.iter().filter(|job| job.unreconciled).count();
        let reusable = job_board.iter().filter(|job| job.reusable_eligible).count();
        println!(
            "job board: {} active, {} unreconciled, {} reusable",
            active, unreconciled, reusable
        );
    }
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
    fn cli_options_default_to_tui() {
        assert_eq!(
            CliOptions::parse_from(std::iter::empty::<&str>()).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Tui
            }
        );
    }

    #[test]
    fn langfuse_startup_toast_reports_enabled_and_missing_config() {
        let enabled = langfuse_startup_toast(&LangfuseTracingStatus::Enabled)
            .expect("enabled should show startup toast");
        assert_eq!(enabled.message(), "Langfuse tracing enabled");
        assert_eq!(enabled.kind(), crate::tui::state::ToastKind::Success);

        let missing = langfuse_startup_toast(&LangfuseTracingStatus::MissingConfig(vec![
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
        ]))
        .expect("missing config should show startup toast");
        assert_eq!(missing.message(), "Langfuse missing: public/secret/host");
        assert_eq!(missing.kind(), crate::tui::state::ToastKind::Error);
    }

    #[test]
    fn cli_options_support_explicit_cli_and_tui() {
        assert_eq!(
            CliOptions::parse_from(["--cli"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: None,
                    json: false
                }
            }
        );
        assert_eq!(
            CliOptions::parse_from(["cli"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: None,
                    json: false
                }
            }
        );
        assert_eq!(
            CliOptions::parse_from(["repl"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: None,
                    json: false
                }
            }
        );
        assert_eq!(
            CliOptions::parse_from(["--tui"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Tui
            }
        );
        assert_eq!(
            CliOptions::parse_from(["tui"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Tui
            }
        );
    }

    #[test]
    fn cli_options_support_prompt_mode() {
        assert_eq!(
            CliOptions::parse_from(["--prompt", "hello"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: Some("hello".into()),
                    json: false
                }
            }
        );
        assert_eq!(
            CliOptions::parse_from(["-p", "hello"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: Some("hello".into()),
                    json: false
                }
            }
        );
        assert_eq!(
            CliOptions::parse_from(["--cli", "--prompt", "hello"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: Some("hello".into()),
                    json: false
                }
            }
        );
    }

    #[test]
    fn cli_options_support_json_flag() {
        assert_eq!(
            CliOptions::parse_from(["--prompt", "hello", "--json"]).unwrap(),
            CliOptions {
                entry_mode: EntryMode::Cli {
                    prompt: Some("hello".into()),
                    json: true
                }
            }
        );
    }

    #[test]
    fn cli_options_reject_invalid_args() {
        assert!(CliOptions::parse_from(["--bogus"]).is_err());
        assert!(CliOptions::parse_from(["--prompt"]).is_err());
        assert!(CliOptions::parse_from(["--prompt", "--json"]).is_err());
        assert!(CliOptions::parse_from(["--json"]).is_err());
    }

    #[test]
    fn repl_parses_help_and_reasoning_commands() {
        assert_eq!(parse_repl_command("/help"), ReplCommand::Help);
        assert_eq!(parse_repl_command("/?"), ReplCommand::Help);
        assert_eq!(parse_repl_command("/reasoning"), ReplCommand::ReasoningShow);
        assert_eq!(
            parse_repl_command("/think high"),
            ReplCommand::ReasoningSet(ModelReasoningEffort::High)
        );
        assert_eq!(
            parse_repl_command("/reasoning off"),
            ReplCommand::ReasoningSet(ModelReasoningEffort::None)
        );
        assert_eq!(parse_repl_command("/new"), ReplCommand::NewSession);
        assert_eq!(parse_repl_command("/compact"), ReplCommand::Compact);
    }

    #[test]
    fn repl_parses_spaced_and_invalid_commands_locally() {
        assert_eq!(
            parse_repl_command("/permission   safe"),
            ReplCommand::PermissionSet(PermissionMode::Safe)
        );
        assert_eq!(
            parse_repl_command("/permission bogus"),
            ReplCommand::Invalid(
                "Unknown permission mode: bogus. Use safe, default, or solo.".into()
            )
        );
        assert_eq!(
            parse_repl_command("/bogus"),
            ReplCommand::Invalid(
                "Unknown command: /bogus. Type /help for available local commands.".into()
            )
        );
        assert_eq!(
            parse_repl_command("/model a b"),
            ReplCommand::Invalid("Usage: /model <id>".into())
        );
        assert_eq!(parse_repl_command("/resume"), ReplCommand::ResumeShow);
    }

    #[test]
    fn repl_subagent_and_child_commands_stay_local() {
        assert_eq!(
            parse_repl_command("@explorer inspect src/main.rs"),
            ReplCommand::Unsupported(
                "CLI does not support @expert delegation yet; use the TUI for subagents.".into()
            )
        );
        assert_eq!(
            parse_repl_command("@fixer wire command parser"),
            ReplCommand::Unsupported(
                "CLI does not support @expert delegation yet; use the TUI for subagents.".into()
            )
        );
        assert_eq!(
            parse_repl_command("/child next"),
            ReplCommand::Unsupported(
                "CLI does not support /child or /children yet; child transcript parity is pending. Use the TUI for child navigation.".into()
            )
        );
        assert_eq!(
            parse_repl_command("/branches"),
            ReplCommand::Unsupported(
                "CLI does not support /branches yet; use the TUI for context branch commands."
                    .into()
            )
        );
        assert_eq!(
            parse_repl_command("/branch feature"),
            ReplCommand::Unsupported(
                "CLI does not support /branch yet; use the TUI for context branch commands.".into()
            )
        );
        assert_eq!(
            parse_repl_command("/checkout feature"),
            ReplCommand::Unsupported(
                "CLI does not support /checkout yet; use the TUI for context branch commands."
                    .into()
            )
        );
        assert_eq!(
            parse_repl_command("/parent"),
            ReplCommand::Unsupported(
                "CLI does not support /parent yet; child transcript parity is pending. Use the TUI for child navigation.".into()
            )
        );
    }
}

fn init_tracing(log_path: &Path) -> TracingGuards {
    let log_file = match open_log_file(log_path) {
        Ok(log_file) => log_file,
        Err(err) => {
            eprintln!(
                "warning: failed to open log file {}: {}; tracing output will not be persisted",
                log_path.display(),
                err
            );
            return TracingGuards::default();
        }
    };

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("letcode=info,async_openai=warn"));
    let (langfuse_provider, langfuse_status) = init_langfuse_provider();
    let langfuse_layer = langfuse_provider.as_ref().map(|provider| {
        let tracer = provider.tracer("letcode-langfuse");
        tracing_opentelemetry::layer()
            .with_tracer(tracer)
            .with_filter(filter::filter_fn(|metadata| {
                metadata.target() == langfuse_trace::TARGET
            }))
    });

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(Mutex::new(log_file))
        .with_target(false)
        .with_ansi(false)
        .compact()
        .with_filter(env_filter);

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(langfuse_layer)
        .init();

    TracingGuards {
        _langfuse: langfuse_provider.map(|provider| LangfuseTracingGuard { provider }),
        langfuse_status,
    }
}

fn init_langfuse_provider() -> (Option<SdkTracerProvider>, LangfuseTracingStatus) {
    match langfuse_env_status() {
        LangfuseEnvStatus::Disabled => return (None, LangfuseTracingStatus::Disabled),
        LangfuseEnvStatus::MissingConfig(missing) => {
            return (None, LangfuseTracingStatus::MissingConfig(missing));
        }
        LangfuseEnvStatus::Ready => {}
    }

    let exporter = match ExporterBuilder::from_env()
        .map(|builder| builder.with_timeout(Duration::from_secs(30)))
        .and_then(|builder| builder.build())
    {
        Ok(exporter) => exporter,
        Err(err) => {
            eprintln!("warning: Langfuse tracing is enabled but could not initialize: {err}");
            return (None, LangfuseTracingStatus::InitializationFailed);
        }
    };

    (
        Some(
            SdkTracerProvider::builder()
                .with_resource(
                    Resource::builder()
                        .with_attributes([
                            KeyValue::new("service.name", "letcode"),
                            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                        ])
                        .build(),
                )
                .with_span_processor(BatchSpanProcessor::builder(exporter, Tokio).build())
                .build(),
        ),
        LangfuseTracingStatus::Enabled,
    )
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn langfuse_env_enabled() -> bool {
    env_flag_enabled("LETCODE_LANGFUSE_ENABLED")
}

fn langfuse_env_status() -> LangfuseEnvStatus {
    if !langfuse_env_enabled() {
        return LangfuseEnvStatus::Disabled;
    }

    let missing = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]
    .into_iter()
    .filter(|name| env_var_missing(name))
    .collect::<Vec<_>>();

    if missing.is_empty() {
        LangfuseEnvStatus::Ready
    } else {
        LangfuseEnvStatus::MissingConfig(missing)
    }
}

fn env_var_missing(name: &str) -> bool {
    env::var(name)
        .map(|value| value.trim().is_empty())
        .unwrap_or(true)
}

fn inspect_langfuse_tracing_status() -> LangfuseTracingStatus {
    match langfuse_env_status() {
        LangfuseEnvStatus::Disabled => LangfuseTracingStatus::Disabled,
        LangfuseEnvStatus::MissingConfig(missing) => LangfuseTracingStatus::MissingConfig(missing),
        LangfuseEnvStatus::Ready => LangfuseTracingStatus::InitializationFailed,
    }
}

fn langfuse_startup_toast(status: &LangfuseTracingStatus) -> Option<tui::StartupToast> {
    match status {
        LangfuseTracingStatus::Disabled => None,
        LangfuseTracingStatus::Enabled => {
            Some(tui::StartupToast::success("Langfuse tracing enabled"))
        }
        LangfuseTracingStatus::MissingConfig(missing) => Some(tui::StartupToast::error(format!(
            "Langfuse missing: {}",
            missing
                .iter()
                .map(|name| short_langfuse_env_label(name))
                .collect::<Vec<_>>()
                .join("/")
        ))),
        LangfuseTracingStatus::InitializationFailed => {
            Some(tui::StartupToast::error("Langfuse init failed; check logs"))
        }
    }
}

fn short_langfuse_env_label(name: &str) -> &'static str {
    match name {
        "LANGFUSE_PUBLIC_KEY" => "public",
        "LANGFUSE_SECRET_KEY" => "secret",
        "LANGFUSE_HOST" => "host",
        _ => "config",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LangfuseEnvStatus {
    Disabled,
    MissingConfig(Vec<&'static str>),
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LangfuseTracingStatus {
    Disabled,
    Enabled,
    MissingConfig(Vec<&'static str>),
    InitializationFailed,
}

struct TracingGuards {
    _langfuse: Option<LangfuseTracingGuard>,
    langfuse_status: LangfuseTracingStatus,
}

impl Default for TracingGuards {
    fn default() -> Self {
        Self {
            _langfuse: None,
            langfuse_status: inspect_langfuse_tracing_status(),
        }
    }
}

struct LangfuseTracingGuard {
    provider: SdkTracerProvider,
}

impl Drop for LangfuseTracingGuard {
    fn drop(&mut self) {
        if let Err(err) = self.provider.shutdown() {
            eprintln!("warning: failed to flush Langfuse traces: {err}");
        }
    }
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
