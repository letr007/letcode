use super::*;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_transcript() -> (std::path::PathBuf, Arc<Mutex<TranscriptRecorder>>, String) {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-session-settings-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript");
    let session_id = recorder.session_id().to_string();
    (base_dir, Arc::new(Mutex::new(recorder)), session_id)
}

fn test_agent() -> Agent {
    Agent::new("gpt-5.5", 1, 1)
}

fn configured_agent(fast_mode_path: &std::path::Path) -> Agent {
    std::fs::write(
        fast_mode_path,
        r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
    )
    .expect("write Fast Mode config");
    let mut agent = test_agent();
    agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));
    let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);
    agent.set_fast_mode(fast_mode);
    agent
}

fn prepared_route(route: ModelRoute) -> crate::agent::PreparedPrimaryRoute {
    crate::agent::PreparedPrimaryRoute::new(
        route,
        crate::config::ApiProtocol::Responses,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        crate::config::RetryConfig::default(),
    )
}

#[test]
fn reasoning_effort_change_is_session_local() {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-session-reasoning-config-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    std::fs::create_dir_all(&base_dir).expect("create config directory");
    let config_path = base_dir.join("letcode.toml");
    std::fs::write(&config_path, "session reasoning sentinel\n").expect("write config");
    let before = std::fs::read_to_string(&config_path).expect("read config before change");
    let mut agent = test_agent();
    agent.set_model_catalog(std::collections::HashMap::from([(
        "gpt-5.5".into(),
        crate::request_builder::ModelRequestMetadata {
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::Medium),
            reasoning_efforts: vec![ModelReasoningEffort::Medium, ModelReasoningEffort::High],
            ..Default::default()
        },
    )]));

    let (_session_dir, transcript, _session_id) = temp_transcript();
    apply_reasoning_effort(&mut agent, &transcript, ModelReasoningEffort::High)
        .expect("reasoning effort changes in memory");

    assert_eq!(agent.reasoning_effort(), Some(ModelReasoningEffort::High));
    assert_eq!(
        agent.model_catalog()["gpt-5.5"].reasoning_effort,
        Some(ModelReasoningEffort::Medium),
        "session selection must not mutate configured model metadata"
    );
    assert_eq!(
        std::fs::read_to_string(&config_path).expect("read config after change"),
        before
    );
    let records = crate::transcript::read_records(transcript.lock().expect("transcript").path())
        .expect("read transcript");
    assert_eq!(
        crate::transcript::restore_latest_reasoning_effort(&records, "gpt-5.5"),
        Some(ModelReasoningEffort::High)
    );
}

#[test]
fn restored_reasoning_effort_is_scoped_to_model_route() {
    let (_base_dir, transcript, _session_id) = temp_transcript();
    {
        let mut recorder = transcript.lock().expect("transcript");
        recorder
            .record_reasoning_effort_changed("primary/gpt-5.5", ModelReasoningEffort::High)
            .expect("record primary effort");
        recorder
            .record_reasoning_effort_changed("other/gpt-5.5", ModelReasoningEffort::Low)
            .expect("record other effort");
    }
    let records = crate::transcript::read_records(transcript.lock().expect("transcript").path())
        .expect("read transcript");
    assert_eq!(
        crate::transcript::restore_latest_reasoning_effort(&records, "primary/gpt-5.5"),
        Some(ModelReasoningEffort::High)
    );
    assert_eq!(
        crate::transcript::restore_latest_reasoning_effort(&records, "other/gpt-5.5"),
        Some(ModelReasoningEffort::Low)
    );
}

#[test]
fn model_route_apply_failure_restores_the_persisted_route() {
    let (_base_dir, transcript, _session_id) = temp_transcript();
    let cloned = Arc::clone(&transcript);
    let join = std::thread::spawn(move || {
        let _guard = cloned.lock().expect("lock transcript");
        panic!("poison transcript mutex for test");
    });
    let _ = join.join();
    let mut agent = test_agent();
    let previous_route = ModelRoute::new("primary", "gpt-5.5");
    let selected_route = ModelRoute::new("expert", "claude-4");
    agent.set_primary_route(previous_route.clone());
    let mut persisted_routes = Vec::new();

    let error = persist_and_apply_model_route_with(
        &mut agent,
        &transcript,
        selected_route.clone(),
        prepared_route(selected_route.clone()),
        |route| {
            persisted_routes.push(route.clone());
            Ok(())
        },
    )
    .expect_err("poisoned transcript must prevent route application");

    assert!(!error.fast_mode_auto_disabled());
    assert_eq!(persisted_routes, [selected_route, previous_route]);
    assert_eq!(agent.model(), "gpt-5.5");
    assert_eq!(
        agent.primary_route(),
        Some(&ModelRoute::new("primary", "gpt-5.5"))
    );
}

#[test]
fn model_route_persistence_failure_leaves_live_agent_unchanged() {
    let (_base_dir, transcript, _session_id) = temp_transcript();
    let cloned = Arc::clone(&transcript);
    let join = std::thread::spawn(move || {
        let _guard = cloned.lock().expect("lock transcript");
        panic!("poison transcript mutex for test");
    });
    let _ = join.join();
    let mut agent = test_agent();
    agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));

    let prepared_route = crate::agent::PreparedPrimaryRoute::new(
        ModelRoute::new("expert", "claude-4"),
        crate::config::ApiProtocol::Responses,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
        crate::config::RetryConfig::default(),
    );
    let error = apply_model_route_with(
        &mut agent,
        &transcript,
        ModelRoute::new("expert", "claude-4"),
        prepared_route,
    )
    .expect_err("poisoned transcript must fail before applying the route");

    assert!(!error.fast_mode_auto_disabled());
    assert_eq!(agent.model(), "gpt-5.5");
    assert_eq!(
        agent.primary_route(),
        Some(&ModelRoute::new("primary", "gpt-5.5"))
    );
}

#[test]
fn model_route_fast_mode_auto_disable_is_memory_only() {
    let (base_dir, transcript, session_id) = temp_transcript();
    let fast_mode_path = base_dir.join("letcode.toml");
    std::fs::write(
        &fast_mode_path,
        r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
    )
    .expect("write Fast Mode config");
    let before = std::fs::read_to_string(&fast_mode_path).expect("read config before apply");
    let mut agent = test_agent();
    agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));
    agent.set_fast_mode(crate::fast_mode::FastMode::load(&fast_mode_path, true));
    let cloned = Arc::clone(&transcript);
    let join = std::thread::spawn(move || {
        let _guard = cloned.lock().expect("lock transcript");
        panic!("poison transcript mutex for test");
    });
    let _ = join.join();

    let error = apply_model_route_with(
        &mut agent,
        &transcript,
        ModelRoute::new("expert", "claude-4"),
        prepared_route(ModelRoute::new("expert", "claude-4")),
    )
    .expect_err("poisoned transcript must prevent route application");

    assert!(error.fast_mode_auto_disabled());
    assert!(!agent.fast_mode_enabled());
    assert_eq!(
        std::fs::read_to_string(&fast_mode_path).expect("read config after apply"),
        before
    );
    assert_eq!(agent.model(), "gpt-5.5");
    assert_eq!(
        agent.primary_route(),
        Some(&ModelRoute::new("primary", "gpt-5.5"))
    );
    assert!(
        crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
            .expect("read records")
            .is_empty()
    );
}

#[test]
fn model_route_transcript_failure_follows_fast_mode_reconciliation_without_route_apply() {
    let (base_dir, transcript, session_id) = temp_transcript();
    let fast_mode_path = base_dir.join("letcode.toml");
    let mut agent = configured_agent(&fast_mode_path);
    let cloned = Arc::clone(&transcript);
    let join = std::thread::spawn(move || {
        let _guard = cloned.lock().expect("lock transcript");
        panic!("poison transcript mutex for test");
    });
    let _ = join.join();

    let error = apply_model_route_with(
        &mut agent,
        &transcript,
        ModelRoute::new("expert", "claude-4"),
        prepared_route(ModelRoute::new("expert", "claude-4")),
    )
    .expect_err("transcript failure must prevent route application");

    assert!(error.fast_mode_auto_disabled());
    assert!(!agent.fast_mode_enabled());
    assert_eq!(agent.model(), "gpt-5.5");
    assert_eq!(
        agent.primary_route(),
        Some(&ModelRoute::new("primary", "gpt-5.5"))
    );
    assert!(
        crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
            .expect("read records")
            .is_empty()
    );
}
