use super::*;
use crate::config::ProviderConfig;
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

fn test_agent() -> Agent {
    Agent::new("test-model", 1, 1)
}

#[test]
fn primary_route_factory_rejects_unknown_provider_or_model() {
    let factory =
        ConfiguredPrimaryRouteFactory::new(IndexMap::new(), config::RetryConfig::default());
    assert!(
        crate::agent::PrimaryRouteFactory::prepare_route(
            &factory,
            config::ModelRoute::new("missing", "shared")
        )
        .is_err()
    );

    let provider = ProviderConfig {
        base_url: "http://127.0.0.1:9876/v1".into(),
        jev_endpoint: None,
        auth_mode: crate::config::ProviderAuthMode::ApiKey,
        api_key: "test-key".into(),
        protocol: config::ApiProtocol::Responses,
        default_model: "available".into(),
        retry: None,
        reviewer: None,
        models: IndexMap::new(),
    };
    let factory = ConfiguredPrimaryRouteFactory::new(
        IndexMap::from([("known".into(), provider)]),
        config::RetryConfig::default(),
    );
    assert!(
        crate::agent::PrimaryRouteFactory::prepare_route(
            &factory,
            config::ModelRoute::new("known", "missing")
        )
        .is_err()
    );
}

mod integration_tests {
    use super::*;

    #[test]
    fn active_provider_model_switch_reconfigures_complete_route() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: config::ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: Some(config::RetryConfig {
                enabled: false,
                max_attempts: 1,
                max_recovery_attempts: 1,
                initial_delay_secs: 1,
                exponential_backoff: false,
                backoff_multiplier: 1.0,
                jitter_secs: 0,
            }),
            reviewer: None,
            models: IndexMap::from([(
                "shared".into(),
                config::ModelConfig {
                    display_name: None,
                    protocol: config::ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    context_window: Some(8_192),
                    effective_input_limit_tokens: Some(4_096),
                    max_output_tokens: Some(512),
                    supports_tools: false,
                    supports_input_images: false,
                    supports_tool_result_images: false,
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
        let records =
            transcript::read_records(recorder.lock().expect("transcript recorder").path())
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
            .record_turn_started(agent::TurnStartedEvent { turn_id: 2 })
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

        let new_session = (|| {
            let prepared = session::prepare_new_session_package(
                &invalid_sessions_dir,
                agent.route_display_name(),
                None,
                &config::SubagentsConfig::default(),
            )?;
            let prepared = session::lifecycle::prepare_new_session_install_with_route(
                &agent, &recorder, prepared, None,
            )?;
            prepared.commit(&mut agent, &recorder);
            Ok::<(), anyhow::Error>(())
        })();
        assert!(new_session.is_err());

        assert_eq!(agent.runtime_snapshot_for_test(), &old_snapshot);
        assert_eq!(
            recorder.lock().expect("recorder poisoned").session_id(),
            old_id
        );
        let _ = fs::remove_dir_all(base_dir);
    }
}

#[test]
fn cli_options_parse_version_and_update_commands() {
    assert_eq!(
        CliOptions::parse_from(["--version"])
            .expect("version should parse")
            .entry_mode,
        EntryMode::Version
    );
    assert_eq!(
        CliOptions::parse_from(["update"])
            .expect("update should parse")
            .entry_mode,
        EntryMode::Update { check_only: false }
    );
    assert_eq!(
        CliOptions::parse_from(["update", "check"])
            .expect("update check should parse")
            .entry_mode,
        EntryMode::Update { check_only: true }
    );
}

#[test]
fn cli_options_parse_config_schema_commands() {
    assert_eq!(
        CliOptions::parse_from(["config", "schema"])
            .expect("config schema should parse")
            .entry_mode,
        EntryMode::GenerateSchema { check: false }
    );
    assert_eq!(
        CliOptions::parse_from(["config", "schema", "check"])
            .expect("config schema check should parse")
            .entry_mode,
        EntryMode::GenerateSchema { check: true }
    );
    assert!(CliOptions::parse_from(["config", "schema", "latest"]).is_err());
    assert!(CliOptions::parse_from(["config", "schema", "check", "extra"]).is_err());
}

#[test]
fn cli_options_reject_invalid_args() {
    assert!(CliOptions::parse_from(["--bogus"]).is_err());
    assert!(CliOptions::parse_from(["--prompt"]).is_err());
    assert!(CliOptions::parse_from(["--prompt", "--json"]).is_err());
    assert!(CliOptions::parse_from(["--json"]).is_err());
    assert!(CliOptions::parse_from(["version", "extra"]).is_err());
    assert!(CliOptions::parse_from(["update", "latest"]).is_err());
    assert!(CliOptions::parse_from(["update", "check", "extra"]).is_err());
}
