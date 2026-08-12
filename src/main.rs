#![cfg_attr(test, allow(unused))]

mod agent;
mod agent_event_journal;
mod cli;
mod code_analysis;
mod command;
mod config;
mod context_tree;
mod context_view;
mod delegation;
mod evidence;
mod fast_mode;
mod langfuse_trace;
mod mcp;
mod memory;
mod permission;
mod protocol_frames;
mod request_builder;
mod retry;
mod runtime_context;
mod session;
mod skills;
mod subagent;
mod subagent_events;
mod tool;
mod tool_format;
mod tool_names;
mod transcript;
mod tui;
mod user_content;

use agent::{Agent, ConfiguredPrimaryRouteFactory, PrimaryRouteFactory as _};
use anyhow::{Result, anyhow, bail};
use async_openai::config::OpenAIConfig;
use config::AppConfig;
use delegation::supported_agent_names;
use fast_mode::FastMode;
use indexmap::IndexMap;
use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_langfuse::ExporterBuilder;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::runtime::Tokio;
use opentelemetry_sdk::trace::{
    SdkTracerProvider, span_processor_with_async_runtime::BatchSpanProcessor,
};
use skills::SkillRegistry;
use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use subagent::ExpertRouteFactory;
use tracing_subscriber::{
    EnvFilter, filter,
    layer::{Layer, SubscriberExt},
    util::SubscriberInitExt,
};
use transcript::{TranscriptRecorder, read_records};
use tui::runtime::{AvailableExpert, AvailableModel};

#[tokio::main]
async fn main() -> Result<()> {
    let options = CliOptions::parse()?;
    dotenvy::dotenv().ok();
    if let EntryMode::ValidateConfig { path } = options.entry_mode {
        return run_config_validate(path);
    }
    let config = AppConfig::load()?;
    let _tracing_guards = init_tracing(&config.global.log_file);

    let (active_provider_name, active_provider) = config.active_provider();
    let active_route = config.active_route();
    let active_provider_label = active_provider_name.to_string();
    let api_key_hint = format!(
        "Set api_key for the selected provider in {}.",
        config.config_path.display(),
    );
    let provider_api_key_hints = config
        .providers
        .keys()
        .map(|provider_name| {
            (
                provider_name.clone(),
                provider_api_key_hint(&config, provider_name),
            )
        })
        .collect::<IndexMap<_, _>>();

    let available_models = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.iter().map(move |(model_id, model)| {
                let route = config::ModelRoute::new(provider_name, model_id);
                AvailableModel::with_context_window_and_reasoning(
                    route.display_name(),
                    provider.model_label(model_id),
                    model.context_window,
                    model.reasoning_effort.clone(),
                    model.request_metadata().selectable_reasoning_efforts(),
                )
            })
        })
        .collect::<Vec<_>>();
    memory::set_memory_sessions_dir(config.global.sessions_dir.clone());
    let client = active_route.build_client(active_provider);
    let mut agent = Agent::new(
        client,
        active_route.model.clone(),
        config.global.max_iterations,
        config.global.max_tool_calls,
    );
    agent.set_fast_mode(FastMode::load(
        &config.config_path,
        config.fast_mode_enabled,
    ));
    agent.auto_disable_fast_mode_for_model(agent.model())?;
    let workspace_dir = env::current_dir()?;
    agent.load_instruction_files_from(&config.config_dir, &workspace_dir)?;
    agent.set_default_protocol(active_provider.protocol);
    let model_catalog = active_provider
        .models
        .iter()
        .map(|(model_id, model)| (model_id.clone(), model.request_metadata()))
        .collect::<HashMap<_, _>>();
    agent.set_model_catalog(model_catalog);
    agent.set_primary_route(active_route.clone());
    let model_protocols = active_provider
        .models
        .iter()
        .map(|(model_id, model)| (model_id.clone(), model.protocol))
        .collect::<HashMap<_, _>>();
    agent.set_model_protocols(model_protocols);
    agent.set_compaction_config(config.global.compaction.clone());
    agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
    agent.set_tool_parallelism(
        config
            .tools
            .parallelism
            .iter()
            .map(|(name, mode)| (name.clone(), *mode)),
    )?;
    agent.set_retry_config(
        active_provider
            .retry
            .clone()
            .unwrap_or_else(|| config.global.retry.clone()),
    );
    agent.set_permission_mode(config.permissions.mode);
    let primary_route_factory = Arc::new(ConfiguredPrimaryRouteFactory::new(
        config.providers.clone(),
        config.global.retry.clone(),
    ));
    let prepared_active_route = primary_route_factory.prepare_route(active_route.clone())?;
    agent.apply_prepared_route(prepared_active_route);
    agent.set_primary_route_factory(primary_route_factory);
    install_expert_route_factory(&mut agent, &config)?;
    let skill_registry = Arc::new(SkillRegistry::load(&config.config_dir, &workspace_dir)?);
    agent.register_skill_registry(skill_registry.clone())?;
    let recorder = Arc::new(Mutex::new(TranscriptRecorder::create(
        &config.global.sessions_dir,
    )?));
    configure_agent_runtime_snapshot_provider(&mut agent, &recorder);
    agent.set_context_scope_state(
        recorder
            .lock()
            .expect("transcript recorder poisoned")
            .context_scope_state(),
    );

    {
        let mut recorder = recorder.lock().expect("transcript recorder poisoned");
        recorder.record_session_started(active_route.display_name())?;
        sync_agent_context_scope_from_recorder(&mut agent, &recorder)?;
    }

    let model_label = active_provider.model_label(&active_route.model);
    let engine_config = session_engine_config(&config, provider_api_key_hints, api_key_hint);
    let initial_reasoning = agent.reasoning_effort();
    let (engine, projection) =
        session::SessionEngine::start(agent, recorder, model_label, engine_config)?;

    match options.entry_mode {
        EntryMode::Cli { prompt, json } => match prompt {
            Some(prompt) => cli::run_one_shot(engine, projection, prompt, json).await?,
            None => {
                cli::run_repl(
                    engine,
                    projection,
                    &config,
                    config.global.sessions_dir.clone(),
                    initial_reasoning,
                )
                .await?;
            }
        },
        mode @ (EntryMode::Tui | EntryMode::Resume { .. }) => {
            let resume_session_id = match mode {
                EntryMode::Resume { session_id } => Some(session_id),
                _ => None,
            };
            tui::run_tui(
                engine,
                projection,
                config.global.sessions_dir.clone(),
                config.config_dir.clone(),
                workspace_dir,
                active_provider_label,
                available_models,
                supported_agent_names()
                    .map(|agent_name| AvailableExpert {
                        agent_name: agent_name.to_string(),
                        route_id: config
                            .expert_route_for(agent_name)
                            .expect("supported expert has a resolved route")
                            .display_name(),
                    })
                    .collect(),
                langfuse_startup_toast(&_tracing_guards.langfuse_status),
                skill_registry.cards(),
                resume_session_id,
            )
            .await?;
        }
        EntryMode::ValidateConfig { .. } => {
            unreachable!("config validate exits before AppConfig::load")
        }
    }

    Ok(())
}

fn session_engine_config(
    config: &AppConfig,
    provider_api_key_hints: IndexMap<String, String>,
    api_key_hint: String,
) -> session::SessionEngineConfig {
    session::SessionEngineConfig {
        sessions_dir: config.global.sessions_dir.clone(),
        model_routes: config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.keys().map(move |model| {
                    let route = config::ModelRoute::new(provider_name, model);
                    (route.display_name(), route)
                })
            })
            .collect(),
        route_api_key_configured: config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.keys().map(move |model| {
                    let route = config::ModelRoute::new(provider_name, model);
                    (route.display_name(), !provider.api_key.trim().is_empty())
                })
            })
            .collect(),
        new_session_default_route: config.active_route(),
        new_session_default_expert_routes: supported_agent_names()
            .filter_map(|agent_name| {
                config
                    .model_route_for(agent_name)
                    .cloned()
                    .map(|route| (agent_name.to_string(), route))
            })
            .collect(),
        expert_model_routes: supported_agent_names()
            .filter_map(|agent_name| {
                config
                    .model_route_for(agent_name)
                    .cloned()
                    .map(|route| (agent_name.to_string(), route))
            })
            .collect(),
        expert_allowed_models: supported_agent_names()
            .map(|agent_name| {
                (
                    agent_name.to_string(),
                    config
                        .agents
                        .allowed_models_for(agent_name)
                        .unwrap_or_default()
                        .to_vec(),
                )
            })
            .collect(),
        legacy_expert_models: supported_agent_names()
            .filter(|agent_name| config.agents.follows_active_provider(agent_name))
            .filter_map(|agent_name| {
                config
                    .model_route_for(agent_name)
                    .map(|route| (agent_name.to_string(), route.model.clone()))
            })
            .collect(),
        providers: config.providers.clone(),
        global_retry: config.global.retry.clone(),
        provider_api_key_hints,
        api_key_hint,
        mcp_config_path: config.config_path.clone(),
        mcp_config: config.mcp.clone(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EntryMode {
    Cli { prompt: Option<String>, json: bool },
    Tui,
    Resume { session_id: String },
    ValidateConfig { path: Option<String> },
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
                "resume" => {
                    let Some(session_id) = args.next() else {
                        bail!("resume requires a session id");
                    };
                    let session_id = session_id.as_ref().to_string();
                    if session_id.is_empty() || session_id.starts_with('-') {
                        bail!("resume requires a session id");
                    }
                    if args.next().is_some() {
                        bail!("resume accepts exactly one session id");
                    }
                    return Ok(Self {
                        entry_mode: EntryMode::Resume { session_id },
                    });
                }
                "config" => {
                    let Some(subcommand) = args.next() else {
                        bail!("config requires a subcommand (validate)");
                    };
                    match subcommand.as_ref() {
                        "validate" => {
                            let path = args.next().map(|value| value.as_ref().to_string());
                            if args.next().is_some() {
                                bail!("config validate accepts at most one path argument");
                            }
                            return Ok(Self {
                                entry_mode: EntryMode::ValidateConfig { path },
                            });
                        }
                        other => bail!("unknown config subcommand: {other}"),
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

fn run_config_validate(path: Option<String>) -> Result<()> {
    let path = match path {
        Some(path) => std::path::PathBuf::from(path),
        None => config::default_config_path()?,
    };
    let report = config::validate_config_file(&path);
    if report.valid {
        println!("valid: {}", report.path);
        if let Some(provider) = &report.active_provider {
            println!("active_provider: {provider}");
        }
        if let Some(route) = &report.active_route {
            println!("active_route: {route}");
        }
        if let Some(mode) = &report.permission_mode {
            println!("permission_mode: {mode}");
        }
        if !report.providers.is_empty() {
            println!("providers: {}", report.providers.join(", "));
        }
        if !report.mcp_servers.is_empty() {
            println!("mcp: {}", report.mcp_servers.join(", "));
        }
        Ok(())
    } else {
        eprintln!("invalid: {}", report.path);
        if let Some(error) = &report.error {
            eprintln!("{error}");
        }
        std::process::exit(1);
    }
}

fn install_expert_route_factory(agent: &mut Agent<OpenAIConfig>, config: &AppConfig) -> Result<()> {
    let policies = supported_agent_names().map(|agent_name| {
        (
            agent_name.to_string(),
            config.model_route_for(agent_name).cloned(),
            config
                .agents
                .allowed_models_for(agent_name)
                .unwrap_or_default()
                .to_vec(),
        )
    });
    let factory =
        ExpertRouteFactory::new_with_policies(policies, &config.providers, &config.global.retry)?;
    agent.set_subagent_child_factory(Arc::new(factory));
    Ok(())
}

fn sync_agent_context_scope_from_recorder<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &TranscriptRecorder,
) -> Result<()> {
    session::sync_agent_context_scope_from_recorder(agent, recorder)
}

fn configure_agent_runtime_snapshot_provider<C: async_openai::config::Config>(
    agent: &mut Agent<C>,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
) {
    let runtime_recorder = Arc::clone(recorder);
    agent.set_runtime_snapshot_provider(Arc::new(move || {
        let recorder = runtime_recorder
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?;
        let records = read_records(recorder.path())?;
        Ok(
            transcript::transcript_projection::project_runtime_restore_snapshot(
                recorder.session_id().to_string(),
                records,
                transcript::transcript_projection::SessionContextCursor {
                    branch_id: recorder.current_context_branch_id().map(str::to_string),
                    leaf_sequence: None,
                },
                &[],
            )?
            .snapshot,
        )
    }));
}

fn provider_api_key_env_var(provider_name: &str) -> String {
    config::provider_api_key_env_var(provider_name)
}

fn provider_api_key_hint(config: &AppConfig, provider_name: &str) -> String {
    format!(
        "Set providers.{provider_name}.api_key in {} or set {}.",
        config.config_path.display(),
        provider_api_key_env_var(provider_name)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use async_openai::Client;
    use serde_json::json;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir() -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let base_dir = std::env::temp_dir().join(format!("letcode-main-test-{timestamp}"));
        fs::create_dir_all(&base_dir).expect("temp dir should be created");
        base_dir
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(
            Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("https://api.openai.com/v1")
                    .with_api_key("test"),
            ),
            "test-model",
            1,
            1,
        )
    }

    #[test]
    fn active_provider_model_switch_reconfigures_complete_route() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            api_key: "expert-key".into(),
            protocol: config::ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: Some(config::RetryConfig {
                enabled: false,
                max_attempts: 1,
                max_recovery_attempts: 1,
                initial_delay_secs: 1,
                backoff_multiplier: 1.0,
                jitter_secs: 0,
            }),
            models: IndexMap::from([(
                "shared".into(),
                config::ModelConfig {
                    display_name: None,
                    protocol: config::ApiProtocol::Completions,
                    context_window: Some(8_192),
                    effective_input_limit_tokens: Some(4_096),
                    max_output_tokens: Some(512),
                    supports_tools: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(test_dir()).expect("create transcript"),
        ));
        let mut agent = test_agent();
        agent.set_primary_route(config::ModelRoute::new("primary", "shared"));

        let factory = Arc::new(ConfiguredPrimaryRouteFactory::new(
            IndexMap::from([("expert".into(), provider)]),
            config::RetryConfig::default(),
        ));
        let route = config::ModelRoute::new("expert", "shared");
        let prepared_route = factory
            .prepare_route(route.clone())
            .expect("route should prepare");
        agent.set_primary_route_factory(factory);
        session::settings::apply_model_route_with(&mut agent, &recorder, route, prepared_route)
            .expect("route switch");

        assert_eq!(agent.model(), "shared");
        assert_eq!(
            agent.default_protocol_for_test(),
            config::ApiProtocol::Completions
        );
        assert_eq!(agent.active_model_metadata().context_window, Some(8_192));
        assert!(!agent.active_model_metadata().supports_tools);
        assert!(!agent.retry_config_for_test().enabled);
        let records = transcript::read_records(
            recorder
                .lock()
                .expect("transcript recorder")
                .path()
                .to_path_buf(),
        )
        .expect("read transcript");
        let model_change = serde_json::to_value(
            records
                .last()
                .expect("provider route switch records provenance"),
        )
        .expect("serialize model change");
        assert_eq!(
            model_change.get("previous_model"),
            Some(&json!("primary/shared"))
        );
        assert_eq!(model_change.get("new_model"), Some(&json!("expert/shared")));
    }

    #[test]
    fn primary_route_factory_rejects_unknown_provider_or_model() {
        let mut agent = test_agent();
        let factory =
            ConfiguredPrimaryRouteFactory::new(IndexMap::new(), config::RetryConfig::default());
        agent.set_primary_route_factory(Arc::new(factory));

        let error = agent
            .switch_primary_route(config::ModelRoute::new("missing", "shared"))
            .expect_err("unknown provider must fail");
        assert!(
            error
                .to_string()
                .contains("provider 'missing' is not defined under [providers]")
        );

        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            api_key: "test-key".into(),
            protocol: config::ApiProtocol::Responses,
            default_model: "available".into(),
            retry: None,
            models: IndexMap::new(),
        };
        let mut agent = test_agent();
        agent.set_primary_route_factory(Arc::new(ConfiguredPrimaryRouteFactory::new(
            IndexMap::from([("known".into(), provider)]),
            config::RetryConfig::default(),
        )));
        let error = agent
            .switch_primary_route(config::ModelRoute::new("known", "missing"))
            .expect_err("unknown model must fail");
        assert!(
            error
                .to_string()
                .contains("model 'missing' is not defined under [providers.known.models]")
        );
    }

    #[test]
    fn resume_session_replaces_the_previous_turn_sequence() {
        let base_dir = test_dir();
        let mut old_recorder = TranscriptRecorder::create(&base_dir).expect("old recorder");
        old_recorder
            .record_session_started("old-model")
            .expect("old session start");
        let recorder = Arc::new(Mutex::new(old_recorder));

        let mut target = TranscriptRecorder::create(&base_dir).expect("target recorder");
        let target_id = target.session_id().to_string();
        target
            .record_session_started("target-model")
            .expect("target session start");
        target
            .record_user_message("target content")
            .expect("target content");
        target
            .record_turn_started(agent::TurnStartedEvent {
                turn_id: 2,
                intent: "test".into(),
                directive: "test".into(),
                validation_reminder: "test".into(),
            })
            .expect("target turn start");
        drop(target);

        let mut agent = test_agent();
        agent
            .restore_session_context(Vec::new(), Vec::new(), 99)
            .expect("seed old turn sequence");

        let prepared = session::prepare_resume_package(&base_dir, target_id.clone())
            .expect("prepare resume target");
        assert_eq!(prepared.snapshot.max_turn_id, 2);
        session::install_prepared_routed_resume_for_agent(&mut agent, &recorder, prepared)
            .expect("resume target");

        assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, None);
        assert_eq!(
            recorder.lock().expect("recorder poisoned").session_id(),
            target_id
        );
        let _ = fs::remove_dir_all(base_dir);
    }

    #[test]
    fn resume_open_failure_leaves_the_live_session_unchanged() {
        let base_dir = test_dir();
        let mut old_recorder = TranscriptRecorder::create(&base_dir).expect("old recorder");
        old_recorder
            .record_session_started("old-model")
            .expect("old session start");
        let old_id = old_recorder.session_id().to_string();
        let recorder = Arc::new(Mutex::new(old_recorder));

        let mut target = TranscriptRecorder::create(&base_dir).expect("target recorder");
        let target_id = target.session_id().to_string();
        target
            .record_session_started("target-model")
            .expect("target session start");
        target
            .record_user_message("target content")
            .expect("target content");
        let target_path = target.path().to_path_buf();
        drop(target);
        let mut file = OpenOptions::new()
            .append(true)
            .open(target_path)
            .expect("target transcript should open");
        writeln!(
            file,
            "{}",
            json!({
                "schema_version": 1,
                "event_id": "uncommitted-test-transaction",
                "scope": "global",
                "base_revision": 2,
                "resulting_revision": 3,
                "transaction_id": "uncommitted-test-transaction",
                "transaction_index": 0,
                "transaction_count": 1,
                "session_id": target_id,
                "sequence": 3,
                "timestamp_ms": 0,
                "event": {"kind": "session_started", "model": "target-model"},
            })
        )
        .expect("uncommitted transaction should be written");

        let mut agent = test_agent();
        agent
            .restore_session_context(Vec::new(), Vec::new(), 99)
            .expect("seed old turn sequence");
        let old_snapshot = agent.runtime_snapshot_for_test().clone();

        let resume = (|| {
            let prepared = session::prepare_resume_package(&base_dir, &target_id)?;
            session::install_prepared_routed_resume_for_agent(&mut agent, &recorder, prepared)?;
            Ok::<(), anyhow::Error>(())
        })();
        assert!(resume.is_err());

        assert_eq!(agent.runtime_snapshot_for_test(), &old_snapshot);
        assert_eq!(
            recorder.lock().expect("recorder poisoned").session_id(),
            old_id
        );
        let _ = fs::remove_dir_all(base_dir);
    }

    #[test]
    fn new_session_creation_failure_does_not_reset_the_live_agent() {
        let base_dir = test_dir();
        let mut old_recorder = TranscriptRecorder::create(&base_dir).expect("old recorder");
        old_recorder
            .record_session_started("old-model")
            .expect("old session start");
        let old_id = old_recorder.session_id().to_string();
        let recorder = Arc::new(Mutex::new(old_recorder));
        let invalid_sessions_dir = base_dir.join("not-a-directory");
        fs::write(&invalid_sessions_dir, "file").expect("blocking file should be written");

        let mut agent = test_agent();
        agent
            .restore_session_context(Vec::new(), Vec::new(), 99)
            .expect("seed old turn sequence");
        let old_snapshot = agent.runtime_snapshot_for_test().clone();

        assert!(
            session::install_new_session_for_agent(&mut agent, &recorder, &invalid_sessions_dir)
                .is_err()
        );

        assert_eq!(agent.runtime_snapshot_for_test(), &old_snapshot);
        assert_eq!(
            recorder.lock().expect("recorder poisoned").session_id(),
            old_id
        );
        let _ = fs::remove_dir_all(base_dir);
    }

    #[test]
    fn cli_options_reject_invalid_args() {
        assert!(CliOptions::parse_from(["--bogus"]).is_err());
        assert!(CliOptions::parse_from(["--prompt"]).is_err());
        assert!(CliOptions::parse_from(["--prompt", "--json"]).is_err());
        assert!(CliOptions::parse_from(["--json"]).is_err());
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
