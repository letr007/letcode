use super::*;
use crate::agent::PrimaryRouteFactory;
use crate::request_builder::ModelReasoningEffort;
use crate::session::runner::{
    clear_live_text_at_child_terminal, clear_live_text_before_assistant_persist,
};
use std::fs;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn reload_for_test(
    agent: &mut Agent,
    engine_config: &mut SessionEngineConfig,
    route_api_key_configured: &mut indexmap::IndexMap<String, bool>,
    provider_api_key_hints: &mut indexmap::IndexMap<String, String>,
    event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) -> Result<()> {
    let config = AppConfig::load_from_path(&engine_config.mcp_config_path)?;
    apply_config_reload(
        agent,
        engine_config,
        config,
        route_api_key_configured,
        provider_api_key_hints,
        event_tx,
    )
}

fn temp_sessions_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "letcode-session-engine-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is valid")
            .as_nanos()
    ))
}

/// Windows 路径里的反斜杠在 TOML 基本字符串中会被当成转义序列，先转义再插值。
fn toml_path(path: &std::path::Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn run_summary(
    run_id: &str,
    child_session_id: &str,
    agent_name: &str,
    status: SubagentStatus,
) -> crate::subagent::SubagentRunSummary {
    crate::subagent::SubagentRunSummary {
        run_id: run_id.into(),
        child_session_id: child_session_id.into(),
        agent_name: agent_name.into(),
        status,
        failure_kind: None,
        summary: String::new(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: status.as_str().into(),
            summary: String::new(),
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: run_id.into(),
            child_session_id: child_session_id.into(),
            raw_excerpt: None,
        },
    }
}

fn active_job(run_id: &str, child_session_id: &str, agent_name: &str) -> SubagentJob {
    SubagentJob {
        active: true,
        run_id: run_id.into(),
        child_session_id: child_session_id.into(),
        agent_name: agent_name.into(),
        status: "running".into(),
        summary: String::new(),
        pool_ordinal: 1,
    }
}

fn child_interrupted(event: SessionTransportEvent) -> Option<String> {
    match event {
        SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            event: SessionEvent::Interrupted,
            ..
        } => Some(child_session_id),
        _ => None,
    }
}

#[test]
fn dropped_child_terminal_only_fires_for_dropped_statuses() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    emit_dropped_child_terminal(
        &tx,
        &run_summary("run", "child", "explorer", SubagentStatus::Completed),
    );
    assert!(rx.try_recv().is_err());
    emit_dropped_child_terminal(
        &tx,
        &run_summary("run", "child", "explorer", SubagentStatus::TimedOut),
    );
    assert_eq!(
        child_interrupted(rx.try_recv().expect("terminal")),
        Some("child".into())
    );
}

#[test]
fn config_templates_keep_windows_paths_parseable() {
    let path =
        std::path::Path::new("C:\\Users\\runneradmin\\AppData\\Local\\Temp\\letcode\\sessions");
    let rendered = format!("sessions_dir = \"{}\"\n", toml_path(path));

    let parsed: toml::Value = toml::from_str(&rendered).expect("windows paths must stay parseable");
    assert_eq!(
        parsed["sessions_dir"].as_str(),
        Some(path.to_string_lossy().as_ref())
    );
}

fn parent_transcript(sessions_dir: &std::path::Path) -> Arc<StdMutex<TranscriptRecorder>> {
    Arc::new(StdMutex::new(
        TranscriptRecorder::create(sessions_dir).expect("create parent transcript"),
    ))
}

/// Agent whose reviewer policy is the given route.
fn reviewer_route_agent(config: &AppConfig, route: ModelRoute) -> Agent {
    let mut agent = Agent::new("current", 1, 1);
    agent.set_subagent_child_factory(Arc::new(
        crate::subagent::ExpertRouteFactory::new_with_policies(
            [("reviewer".to_string(), Some(route), Vec::new(), None)],
            &config.providers,
            &config.global.retry,
        )
        .expect("reviewer route factory"),
    ));
    agent
}

fn add_child(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: &std::path::Path,
    run_id: &str,
    pool_ordinal: u32,
) -> String {
    let child_dir = crate::transcript::child_sessions_dir(sessions_dir);
    let mut child = TranscriptRecorder::create(child_dir).expect("create child transcript");
    let child_session_id = child.session_id().to_string();
    child
        .record_user_message("child transcript")
        .expect("record child message");
    drop(child);

    let mut parent = transcript.lock().expect("parent transcript");
    let parent_session_id = parent.session_id().to_string();
    parent
        .record_subagent_started(
            run_id,
            parent_session_id,
            "turn-1",
            &child_session_id,
            "explorer",
            "inspect child",
            pool_ordinal,
        )
        .expect("record child start");
    child_session_id
}

fn child_view_event(event: SessionTransportEvent) -> (String, usize, usize) {
    match event {
        SessionTransportEvent::ChildSessionViewed {
            child_session_id,
            index,
            total,
            ..
        } => (child_session_id, index, total),
        other => panic!("expected child session view, got {other:?}"),
    }
}

fn child_terminal(child_session_id: &str) -> SessionTransportEvent {
    SessionTransportEvent::ChildSessionEvent {
        child_session_id: child_session_id.into(),
        agent_name: Some("explorer".into()),
        parent_tool_call_id: None,
        event: SessionEvent::AssistantDone { message_id: None },
    }
}

/// The producer drops a child's streaming tail when its turn ends, so the
/// queued drain forwards the terminal event with no live text behind it.
#[tokio::test]
async fn queued_terminal_child_events_leave_no_live_text() {
    let live_text = crate::session::child_view::ChildLiveText::default();
    live_text.append("child-1", "partial");
    assert_eq!(live_text.get("child-1").as_deref(), Some("partial"));
    clear_live_text_at_child_terminal(
        &live_text,
        "child-1",
        &SessionTransportEvent::AssistantDone { message_id: None },
    );
    assert_eq!(
        live_text.get("child-1"),
        None,
        "the producer clears the tail when the child's turn ends"
    );

    let (runner_tx, mut runner_rx) = mpsc::unbounded_channel();
    let (session_tx, mut session_rx) = mpsc::unbounded_channel();
    runner_tx
        .send(child_terminal("child-1"))
        .expect("queue the terminal child event");

    forward_queued_runner_events(&mut runner_rx, &session_tx);
    assert!(
        session_rx.try_recv().is_ok(),
        "the queued child event is forwarded"
    );
    assert_eq!(
        live_text.get("child-1"),
        None,
        "a forwarded terminal child event must not leave a streaming tail"
    );
}

/// Persisting an assistant message drops its live prefix, so a turn that
/// continues without a tool boundary keeps only the unpersisted tail: a
/// projection reading the journal plus live text sees the text once.
#[test]
fn persisting_an_assistant_message_drops_its_live_prefix_before_the_continuation() {
    let live_text = crate::session::child_view::ChildLiveText::default();
    live_text.append("child-1", "persisted");
    clear_live_text_before_assistant_persist(
        &live_text,
        "child-1",
        &AgentEvent::AssistantMessage {
            content: "persisted".into(),
        },
    );
    assert_eq!(
        live_text.get("child-1"),
        None,
        "the durable text must leave the live buffer at persistence"
    );
    live_text.append("child-1", "continued");

    assert_eq!(
        live_text.get("child-1").as_deref(),
        Some("continued"),
        "the persisted segment must not remain in the in-progress text"
    );
}

#[tokio::test]
async fn command_ingress_preserves_fifo_order() {
    let (mut engine, ingress, _egress) = SessionEngine::new();
    ingress
        .submit(SessionCommand::SetModel("first".into()))
        .expect("engine accepts first command");
    ingress
        .submit(SessionCommand::SetModel("second".into()))
        .expect("engine accepts second command");

    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Command(SessionEngineCommand::SetModel(model))) if model == "first"
    ));
    assert!(matches!(
        engine.recv_control().await,
        Some(SessionEngineControl::Command(SessionEngineCommand::SetModel(model))) if model == "second"
    ));
}

#[test]
fn missing_api_key_error_emits_error_then_exactly_one_done() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    send_missing_api_key_error(
        &tx,
        "expert/shared",
        &indexmap::IndexMap::from([("expert".into(), "Set EXPERT_API_KEY.".into())]),
        "fallback",
    );

    assert!(matches!(
        rx.try_recv(),
        Ok(SessionTransportEvent::Error(error))
            if error.message == "API key is not set for the selected provider. Set EXPERT_API_KEY."
    ));
    assert!(matches!(rx.try_recv(), Ok(SessionTransportEvent::Done)));
    assert!(rx.try_recv().is_err());
}

fn test_engine_config(
    mcp_config_path: std::path::PathBuf,
    model_routes: indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: indexmap::IndexMap<String, bool>,
    expert_default_routes: indexmap::IndexMap<String, ModelRoute>,
    expert_allowed_models: indexmap::IndexMap<String, Vec<ModelRoute>>,
    expert_reasoning_efforts: indexmap::IndexMap<String, ModelReasoningEffort>,
    providers: indexmap::IndexMap<String, crate::config::ProviderConfig>,
    global_retry: crate::config::RetryConfig,
    provider_api_key_hints: indexmap::IndexMap<String, String>,
    new_session_default_route: ModelRoute,
    runtime_catalog: crate::model_runtime::ResolvedRuntimeCatalog,
) -> SessionEngineConfig {
    SessionEngineConfig {
        sessions_dir: std::path::PathBuf::new(),
        workspace_dir: None,
        submit_while_running: crate::config::SubmitWhileRunning::default(),
        model_routes,
        route_api_key_configured,
        new_session_default_route,
        new_session_subagents_policy: crate::config::SubagentsConfig::default(),
        expert_default_routes,
        expert_allowed_models,
        expert_reasoning_efforts,
        providers,
        global_retry,
        provider_api_key_hints,
        api_key_hint: String::new(),
        mcp_config_path,
        mcp_config: indexmap::IndexMap::new(),
        runtime_catalog,
    }
}

const STUB_ANSWER: &str = "the session is still usable";

struct HoldingTool {
    release: Option<Arc<tokio::sync::Notify>>,
}

#[async_trait::async_trait]
impl ToolHandler for HoldingTool {
    fn name(&self) -> &str {
        "test__hold"
    }

    fn description(&self) -> &str {
        "Holds an assistant tool call batch open"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    fn permission_class(&self) -> crate::permission::ToolPermissionClass {
        crate::permission::ToolPermissionClass::Read
    }

    async fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value> {
        match &self.release {
            Some(release) => release.notified().await,
            None => std::future::pending().await,
        }
        Ok(json!({ "released": true }))
    }
}

async fn serve_responses_stub(listener: tokio::net::TcpListener) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut served = 0usize;
    while let Ok(Ok((mut stream, _))) =
        tokio::time::timeout(Duration::from_secs(5), listener.accept()).await
    {
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        let (header_end, content_length) = loop {
            let read = stream.read(&mut chunk).await.expect("stub request read");
            assert!(read > 0, "stub request closed before headers");
            request.extend_from_slice(&chunk[..read]);
            let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let header_end = at + 4;
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .expect("responses request has content length");
            break (header_end, content_length);
        };
        while request.len() < header_end + content_length {
            let read = stream.read(&mut chunk).await.expect("stub body read");
            assert!(read > 0, "stub request closed before body");
            request.extend_from_slice(&chunk[..read]);
        }

        let body = if served == 0 {
            stub_tool_call_sse()
        } else {
            stub_text_sse()
        };
        served += 1;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("stub response write");
    }
}

fn stub_tool_call_sse() -> String {
    [
        json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"test__hold"}}),
        json!({"type":"response.function_call_arguments.done","item_id":"item-1","arguments":"{}"}),
        json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"test__hold","arguments":"{}"}}),
        json!({"type":"response.completed","response":{"id":"resp-1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}),
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect()
}

fn stub_text_sse() -> String {
    [
        json!({"type":"response.output_text.delta","delta":STUB_ANSWER}),
        json!({"type":"response.completed","response":{"id":"resp-2","status":"completed","usage":{"input_tokens":12,"output_tokens":6,"total_tokens":18}}}),
    ]
    .iter()
    .map(|event| format!("data: {event}\n\n"))
    .collect()
}

fn open_tool_call_batch_recorded(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Result<bool> {
    let path = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();
    let records = read_records(&path)?;
    Ok(records.iter().any(|record| {
        matches!(
            &record.event,
            TranscriptEvent::AssistantTurn(turn) if !turn.calls.is_empty()
        )
    }))
}

fn observed_error(error: &ErrorEvent) -> String {
    match error.details.as_deref().filter(|detail| !detail.is_empty()) {
        Some(detail) => format!("{} ({detail})", error.message),
        None => error.message.clone(),
    }
}

mod integration_tests {
    use super::*;
    include!("messaging_tests.rs");

    #[test]
    fn dropped_runs_emit_one_child_terminal_each_and_skip_finished_runs() {
        let runtime = SubagentPool::new();
        runtime.record_completed_for_test(run_summary(
            "run-cancelled",
            "child-cancelled",
            "explorer",
            SubagentStatus::Cancelled,
        ));
        runtime.record_completed_for_test(run_summary(
            "run-timed-out",
            "child-timed-out",
            "explorer",
            SubagentStatus::TimedOut,
        ));
        runtime.record_completed_for_test(run_summary(
            "run-completed",
            "child-completed",
            "explorer",
            SubagentStatus::Completed,
        ));
        runtime.record_completed_for_test(run_summary(
            "run-historian",
            "child-historian",
            "historian",
            SubagentStatus::Cancelled,
        ));
        let runs = vec![
            active_job("run-cancelled", "child-cancelled", "explorer"),
            active_job("run-timed-out", "child-timed-out", "explorer"),
            active_job("run-completed", "child-completed", "explorer"),
            active_job("run-historian", "child-historian", "historian"),
        ];

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut notified = std::collections::HashMap::new();
        emit_dropped_child_terminals(&tx, &runtime, &runs, &mut notified);

        let mut children = Vec::new();
        while let Ok(event) = rx.try_recv() {
            children.push(child_interrupted(event).expect("only child terminal events"));
        }
        children.sort();
        assert_eq!(children, vec!["child-cancelled", "child-timed-out"]);

        emit_dropped_child_terminals(&tx, &runtime, &runs, &mut notified);
        assert!(
            rx.try_recv().is_err(),
            "a run must not receive a second terminal event"
        );
    }

    #[test]
    fn interrupting_one_run_does_not_terminalize_an_unrelated_running_child() {
        let runtime = SubagentPool::new();
        runtime.record_completed_for_test(run_summary(
            "run-cancelled",
            "child-cancelled",
            "explorer",
            SubagentStatus::Cancelled,
        ));
        let runs = vec![
            active_job("run-other", "child-other", "explorer"),
            active_job("run-cancelled", "child-cancelled", "explorer"),
        ];

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut notified = std::collections::HashMap::new();
        emit_dropped_child_terminals(&tx, &runtime, &runs, &mut notified);
        assert_eq!(
            child_interrupted(rx.try_recv().expect("terminal")),
            Some("child-cancelled".into())
        );
        assert!(
            rx.try_recv().is_err(),
            "the still-running child stays untouched"
        );

        send_subagent_interrupted(&tx);
        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::Interrupted)
        ));
        assert!(
            rx.try_recv().is_err(),
            "the parent interrupt carries no child-scope event"
        );
    }

    #[test]
    fn cancelled_background_completion_emits_a_terminal_without_continuation() {
        let runtime = SubagentPool::new();
        let transcript = parent_transcript(&temp_sessions_dir());
        let parent_session_id = transcript
            .lock()
            .expect("lock transcript")
            .session_id()
            .to_string();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut notified = std::collections::HashMap::new();
        let result = Ok(run_summary(
            "run-cancelled",
            "child-cancelled",
            "explorer",
            SubagentStatus::Cancelled,
        ));

        let deliverable = observe_background_subagent_completion(
            &transcript,
            &runtime,
            &mut notified,
            &tx,
            &parent_session_id,
            None,
            &result,
        );
        assert!(
            !deliverable,
            "a cancelled background run must not continue the turn"
        );
        assert_eq!(
            child_interrupted(rx.try_recv().expect("terminal")),
            Some("child-cancelled".into())
        );
        assert!(
            rx.try_recv().is_err(),
            "no background completion event is delivered for a cancelled run"
        );
    }

    #[test]
    fn timed_out_background_completion_emits_a_terminal_and_still_delivers() {
        let runtime = SubagentPool::new();
        let transcript = parent_transcript(&temp_sessions_dir());
        let parent_session_id = transcript
            .lock()
            .expect("lock transcript")
            .session_id()
            .to_string();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut notified = std::collections::HashMap::new();
        let result = Ok(run_summary(
            "run-timed-out",
            "child-timed-out",
            "explorer",
            SubagentStatus::TimedOut,
        ));

        let deliverable = observe_background_subagent_completion(
            &transcript,
            &runtime,
            &mut notified,
            &tx,
            &parent_session_id,
            None,
            &result,
        );
        assert!(deliverable);
        assert_eq!(
            child_interrupted(rx.try_recv().expect("terminal")),
            Some("child-timed-out".into())
        );
        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::BackgroundSubagentCompleted { .. })
        ));
    }

    #[test]
    fn completed_background_run_emits_no_terminal() {
        let runtime = SubagentPool::new();
        let transcript = parent_transcript(&temp_sessions_dir());
        let parent_session_id = transcript
            .lock()
            .expect("lock transcript")
            .session_id()
            .to_string();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut notified = std::collections::HashMap::new();
        let result = Ok(run_summary(
            "run-completed",
            "child-completed",
            "explorer",
            SubagentStatus::Completed,
        ));

        assert!(observe_background_subagent_completion(
            &transcript,
            &runtime,
            &mut notified,
            &tx,
            &parent_session_id,
            None,
            &result,
        ));
        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::BackgroundSubagentCompleted { .. })
        ));
        assert!(
            rx.try_recv().is_err(),
            "a completed run already published its terminal"
        );
    }

    #[test]
    fn reviewer_route_selects_the_backend_from_its_provider() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("letcode.toml");
        fs::write(
            &config_path,
            r#"
active_provider = "chat"
[providers.chat]
protocol = "responses"
default_model = "current"
[providers.chat.auth]
type = "bearer"
credential = "chat-key"
[providers.chat.endpoints]
base_url = "http://127.0.0.1:1"
[providers.chat.models.current]

[providers.typesafe]
protocol = "responses"
default_model = "jev-latest"
reviewer = "jev"
[providers.typesafe.auth]
type = "bearer"
credential = "typesafe-key"
[providers.typesafe.endpoints]
base_url = "https://api.typesafe.ai"
[providers.typesafe.models."jev-latest"]
"#,
        )
        .unwrap();
        let config = AppConfig::load_from_path(&config_path).expect("config");

        let marked = reviewer_route_agent(&config, ModelRoute::new("typesafe", "jev-latest"));
        let jev = reviewer_jev_config(&marked, &config.providers).expect("jev backend");
        assert_eq!(jev.provider, "typesafe");
        assert_eq!(jev.base_url, "https://api.typesafe.ai");
        assert_eq!(jev.model, "jev-latest");
        assert_eq!(jev.credential, "typesafe-key");

        let unmarked = reviewer_route_agent(&config, ModelRoute::new("chat", "current"));
        assert!(reviewer_jev_config(&unmarked, &config.providers).is_none());
    }

    #[tokio::test]
    async fn new_session_inherits_current_model_expert_fake_and_reasoning_settings() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "default"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.current]
[providers.test.models.current.capabilities]
reasoning = true
generation = { reasoning = true }
[providers.test.models.current.generation]
reasoning_efforts = ["high"]
[providers.test.models.default]
[providers.test.models.expert]
[subagents]
mode = "swarm"
max_depth = 4
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let current_route = ModelRoute::new("test", "current");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("current", 1, 1);
            agent.apply_prepared_route(
                primary_factory
                    .prepare_route(current_route.clone())
                    .unwrap(),
            );
            agent.set_primary_route_factory(primary_factory);
            agent.set_fake_client(Some(crate::fake::FakeClient::Codex));
            agent
                .set_reasoning_effort(ModelReasoningEffort::High)
                .unwrap();
            let transcript = parent_transcript(&config.global.sessions_dir);
            transcript
                .lock()
                .unwrap()
                .record_session_started(current_route.display_name())
                .unwrap();
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let (mut engine, _) = SessionEngine::start(
                agent,
                transcript.clone(),
                "Current".into(),
                SessionEngineConfig {
                    sessions_dir: config.global.sessions_dir.clone(),
                    workspace_dir: None,
                    submit_while_running: crate::config::SubmitWhileRunning::default(),
                    model_routes: indexmap::IndexMap::from([
                        ("test/current".into(), current_route.clone()),
                        ("test/default".into(), ModelRoute::new("test", "default")),
                    ]),
                    route_api_key_configured: indexmap::IndexMap::from([
                        ("test/current".into(), true),
                        ("test/default".into(), true),
                        ("test/expert".into(), true),
                    ]),
                    new_session_default_route: ModelRoute::new("test", "default"),
                    new_session_subagents_policy: config.subagents.clone(),
                    expert_default_routes: indexmap::IndexMap::from([
                        ("explorer".into(), ModelRoute::new("test", "default")),
                        ("reviewer".into(), ModelRoute::new("test", "default")),
                    ]),
                    expert_allowed_models: indexmap::IndexMap::new(),
                    expert_reasoning_efforts: indexmap::IndexMap::new(),
                    providers: config.providers.clone(),
                    global_retry: config.global.retry.clone(),
                    provider_api_key_hints: indexmap::IndexMap::new(),
                    api_key_hint: String::new(),
                    mcp_config_path: config.config_path.clone(),
                    mcp_config: config.mcp.clone(),
                    runtime_catalog: config.runtime_catalog.clone(),
                },
            )
            .unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            ingress.submit(SessionCommand::NewSession).unwrap();

            let mut started = None;
            let mut fake = None;
            let mut effort = None;
            while started.is_none() || fake.is_none() || effort.is_none() {
                match tokio::time::timeout(Duration::from_secs(5), events.recv())
                    .await
                    .unwrap()
                    .unwrap()
                {
                    SessionTransportEvent::SessionStarted { expert_models, .. } => {
                        started = Some(expert_models)
                    }
                    SessionTransportEvent::FakeClientChanged { client } => fake = Some(client),
                    SessionTransportEvent::ReasoningEffortChanged { effort: value } => {
                        effort = Some(value)
                    }
                    _ => {}
                }
            }
            let started = started.unwrap();
            assert_eq!(started.get("explorer").unwrap(), "test/default");
            assert_eq!(started.get("reviewer").unwrap(), "test/default");
            assert_eq!(fake, Some(Some(crate::fake::FakeClient::Codex)));
            assert_eq!(effort, Some(ModelReasoningEffort::High));

            let records =
                crate::transcript::read_records(transcript.lock().unwrap().path()).unwrap();
            assert_eq!(
                crate::transcript::restore_subagents_policy(&records).unwrap(),
                config.subagents
            );
            assert_eq!(
                crate::transcript::restore_latest_fake_client(&records),
                Some(crate::fake::FakeClient::Codex)
            );
            assert_eq!(
                crate::transcript::restore_latest_reasoning_effort(&records, "test/current",),
                Some(ModelReasoningEffort::High)
            );
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("new-session inheritance timed out");
    }

    #[tokio::test]
    async fn fresh_session_can_prepare_history_without_a_restore() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let mut recorder = TranscriptRecorder::create(&config.global.sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));
            let mut agent = Agent::new("model", 1, 1);
            agent.set_primary_route(ModelRoute::new("test", "model"));
            agent.set_resolved_model_route(Some(Arc::new(
                config
                    .runtime_catalog
                    .route("test", "model")
                    .unwrap()
                    .clone(),
            )));
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript, "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            ingress.submit(SessionCommand::Compact).unwrap();
            loop {
                match events.recv().await.expect("engine event stream") {
                    SessionTransportEvent::CompactionNoProgress { .. } => break,
                    event @ (SessionTransportEvent::CompactionFailed
                    | SessionTransportEvent::Error(_)) => {
                        panic!("empty history must report no progress: {event:?}");
                    }
                    _ => {}
                }
            }
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("new-session history preparation timed out");
    }

    #[tokio::test]
    async fn interrupted_manual_compaction_reports_cancelled() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let mut recorder = TranscriptRecorder::create(&config.global.sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));
            let mut agent = Agent::new("model", 1, 1);
            agent.set_primary_route(ModelRoute::new("test", "model"));
            agent.set_resolved_model_route(Some(Arc::new(
                config
                    .runtime_catalog
                    .route("test", "model")
                    .unwrap()
                    .clone(),
            )));
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript, "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            ingress.submit(SessionCommand::Compact).unwrap();
            ingress.submit(SessionCommand::Interrupt).unwrap();

            let mut cancelled = false;
            loop {
                match events.recv().await.expect("engine event stream") {
                    SessionTransportEvent::CompactionCancelled => cancelled = true,
                    SessionTransportEvent::CompactionFailed => {
                        panic!("interrupted manual compaction must report cancellation")
                    }
                    SessionTransportEvent::Done => break,
                    _ => {}
                }
            }
            assert!(cancelled, "interrupt must emit CompactionCancelled");
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("interrupted manual compaction timed out");
    }

    #[tokio::test]
    async fn resume_reports_the_recorded_permission_mode_before_session_resumed() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let sessions_dir = config.global.sessions_dir.clone();

            let mut resumed = TranscriptRecorder::create(&sessions_dir).unwrap();
            resumed.record_session_started("test/model").unwrap();
            resumed
                .record_permission_mode_changed("default", "yolo")
                .unwrap();
            let resumed_session_id = resumed.session_id().to_string();
            drop(resumed);

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));
            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", 1, 1);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            // The live process runs in a different mode than the resumed session
            // recorded, so the announced mode can only come from the transcript.
            assert_eq!(agent.permission_mode().to_string(), "default");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript, "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            ingress
                .submit_transitional(SessionEngineCommand::ResumeSession(
                    resumed_session_id.clone(),
                ))
                .unwrap();

            let mut mode_index = None;
            let mut resumed_index = None;
            let mut index = 0;
            while resumed_index.is_none() {
                let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
                    .await
                    .expect("resume must report the resumed session")
                    .expect("engine event stream");
                match event {
                    SessionTransportEvent::PermissionModeChanged { mode } => {
                        assert_eq!(mode, "yolo", "restored mode is announced as-is");
                        mode_index = Some(index);
                    }
                    SessionTransportEvent::SessionResumed { session_id, .. } => {
                        assert_eq!(session_id, resumed_session_id);
                        resumed_index = Some(index);
                    }
                    SessionTransportEvent::Error(error) => panic!("resume failed: {error:?}"),
                    _ => {}
                }
                index += 1;
            }
            let mode_index = mode_index.expect("the recorded permission mode is reported");
            let resumed_index = resumed_index.expect("SessionResumed was reported");
            assert!(
                mode_index < resumed_index,
                "the restored mode must precede SessionResumed: mode at {mode_index}, resumed at {resumed_index}"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("resume of a recorded permission mode timed out");
    }

    #[tokio::test]
    async fn child_view_refresh_skips_unchanged_journals_and_follows_the_moving_ones() {
        let sessions_dir = temp_sessions_dir();
        let parent = parent_transcript(&sessions_dir);
        add_child(&parent, &sessions_dir, "run-1", 1);
        let visible_child = add_child(&parent, &sessions_dir, "run-2", 2);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut visible_child_session_id = Some(visible_child.clone());
        let mut view_state = None;
        let mut cache = None;

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &crate::session::child_view::ChildLiveText::default(),
        )
        .await;
        assert_eq!(
            child_view_event(rx.try_recv().expect("the first pass reports the view")),
            (visible_child.clone(), 1, 2)
        );
        assert!(
            cache.is_some(),
            "a completed pass is kept for the next tick"
        );

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &crate::session::child_view::ChildLiveText::default(),
        )
        .await;
        assert!(
            rx.try_recv().is_err(),
            "journals that have not moved must not be reported again"
        );

        let mut child = TranscriptRecorder::open(
            crate::transcript::child_sessions_dir(&sessions_dir),
            visible_child.clone(),
        )
        .expect("open the visible child transcript");
        child
            .record_user_message("more child work")
            .expect("record the new child message");
        drop(child);

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &crate::session::child_view::ChildLiveText::default(),
        )
        .await;
        let (_, index, total) =
            child_view_event(rx.try_recv().expect("a grown child journal is reported"));
        assert_eq!((index, total), (1, 2), "the sibling count has not changed");

        add_child(&parent, &sessions_dir, "run-3", 3);
        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &crate::session::child_view::ChildLiveText::default(),
        )
        .await;
        let (child_session_id, index, total) = child_view_event(
            rx.try_recv()
                .expect("a moved parent journal is re-resolved"),
        );
        assert_eq!(
            (child_session_id.as_str(), index, total),
            (visible_child.as_str(), 1, 3)
        );

        view_state = None;
        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &crate::session::child_view::ChildLiveText::default(),
        )
        .await;
        assert_eq!(
            child_view_event(
                rx.try_recv()
                    .expect("a cleared view state is answered with a full pass")
            )
            .2,
            3
        );

        let root = parent_transcript(&sessions_dir);
        root.lock()
            .unwrap()
            .record_subagents_policy(&crate::config::SubagentsConfig {
                mode: crate::config::SubagentMode::Swarm,
                ..Default::default()
            })
            .unwrap();
        let root_id = root.lock().unwrap().session_id().to_string();
        let member = |parent: &Arc<StdMutex<TranscriptRecorder>>, run_id: &str, ordinal, depth| {
            let parent_id = parent.lock().unwrap().session_id().to_string();
            let child_id = add_child(parent, &sessions_dir, run_id, ordinal);
            let mut child = TranscriptRecorder::open(
                crate::transcript::child_sessions_dir(&sessions_dir),
                child_id.clone(),
            )
            .unwrap();
            child
                .record_subagent_started(
                    run_id,
                    &parent_id,
                    "turn-1",
                    &child_id,
                    "explorer",
                    "inspect child",
                    ordinal,
                )
                .unwrap();
            child
                .record_subagent_activation(
                    run_id.into(),
                    crate::subagent::SwarmMember {
                        root_session_id: root_id.clone(),
                        parent_session_id: parent_id,
                        child_session_id: child_id,
                        depth,
                        agent_name: "explorer".into(),
                        pool_ordinal: ordinal,
                    },
                )
                .unwrap();
            Arc::new(StdMutex::new(child))
        };
        let branch = member(&root, "swarm-1", 1, 1);
        let visible = member(&root, "swarm-2", 2, 1);
        let visible_id = visible.lock().unwrap().session_id().to_string();
        visible_child_session_id = Some(visible_id.clone());
        view_state = None;
        cache = None;
        let live_text = crate::session::child_view::ChildLiveText::default();
        for _ in 0..2 {
            refresh_visible_child_session_view(
                &root,
                &tx,
                &sessions_dir,
                &mut visible_child_session_id,
                &mut view_state,
                &mut cache,
                &live_text,
            )
            .await;
        }
        assert_eq!(
            child_view_event(rx.try_recv().unwrap()),
            (visible_id.clone(), 1, 2)
        );
        assert!(rx.try_recv().is_err());
        let root_fingerprint = journal_fingerprint(root.lock().unwrap().path());
        let visible_fingerprint = journal_fingerprint(visible.lock().unwrap().path());
        let nested = member(&branch, "swarm-3", 3, 2);
        let deepest = member(&nested, "swarm-4", 4, 3);
        let deepest_id = deepest.lock().unwrap().session_id().to_string();
        assert_eq!(
            journal_fingerprint(root.lock().unwrap().path()),
            root_fingerprint
        );
        assert_eq!(
            journal_fingerprint(visible.lock().unwrap().path()),
            visible_fingerprint
        );
        refresh_visible_child_session_view(
            &root,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
            &live_text,
        )
        .await;
        assert_eq!(child_view_event(rx.try_recv().unwrap()), (visible_id, 1, 4));
        let members =
            crate::session::child_view::project_session_members(&sessions_dir, &root_id).unwrap();
        assert_eq!(
            members
                .iter()
                .map(|member| member.pool_ordinal)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let crate::command::CommandIntent::Child(navigation) =
            crate::command::parse_command(&format!("/child {deepest_id}")).unwrap()
        else {
            panic!("expected child navigation");
        };
        apply_engine_view_child(
            &root,
            &tx,
            Some(&sessions_dir),
            navigation,
            None,
            &mut visible_child_session_id,
            &mut view_state,
            &live_text,
        );
        assert_eq!(
            child_view_event(rx.try_recv().unwrap()),
            (deepest_id.clone(), 3, 4)
        );
        assert_eq!(
            visible_child_session_id.as_deref(),
            Some(deepest_id.as_str())
        );
        let view = crate::session::child_view::project_child_session_view_from_file(
            &sessions_dir,
            &root_id,
            crate::command::ChildNavigation::Prev,
            Some(&deepest_id),
            &live_text,
        )
        .unwrap()
        .unwrap();
        assert_eq!((view.pool_ordinal, view.index, view.total), (3, 2, 4));
    }

    #[test]
    fn subagents_policy_reload_updates_new_sessions_and_keeps_active_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("letcode.toml");
        fs::write(
            &path,
            r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1/v1"
[providers.test.models.model]
"#,
        )
        .unwrap();
        let initial = AppConfig::load_from_path(&path).unwrap();
        let mut engine_config =
            crate::session_engine_config(&initial, Default::default(), String::new(), None);
        let mut agent = Agent::new("model", 1, 1);
        agent.set_primary_route(initial.active_route());
        let mut keys = std::mem::take(&mut engine_config.route_api_key_configured);
        let mut hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        let (tx, mut rx) = mpsc::unbounded_channel();
        apply_config_reload(
            &mut agent,
            &mut engine_config,
            initial.clone(),
            &mut keys,
            &mut hints,
            &tx,
        )
        .unwrap();
        while rx.try_recv().is_ok() {}
        let prepared = crate::session::prepare_new_session_package(
            dir.path(),
            agent.route_display_name(),
            None,
            &initial.subagents,
        )
        .unwrap();
        let original_id = prepared.session_id.clone();
        let live = Arc::new(StdMutex::new(prepared.recorder));
        live.lock()
            .unwrap()
            .record_user_message("keep original session")
            .unwrap();
        let mut changed = initial;
        changed.subagents = crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_depth: 6,
            run_timeout_secs: 20,
            ..Default::default()
        };
        let expected = changed.subagents.clone();
        apply_config_reload(
            &mut agent,
            &mut engine_config,
            changed.clone(),
            &mut keys,
            &mut hints,
            &tx,
        )
        .unwrap();
        assert_eq!(
            agent.subagents_policy(),
            &crate::config::SubagentsConfig::default()
        );
        assert_eq!(engine_config.new_session_subagents_policy, expected);
        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::ModelCatalogUpdated(_))
        ));
        assert!(rx.try_recv().is_err());
        apply_config_reload(
            &mut agent,
            &mut engine_config,
            changed,
            &mut keys,
            &mut hints,
            &tx,
        )
        .unwrap();
        assert!(rx.try_recv().is_err());
        let prepared = crate::session::prepare_new_session_package(
            dir.path(),
            agent.route_display_name(),
            None,
            &engine_config.new_session_subagents_policy,
        )
        .unwrap();
        assert_eq!(
            crate::transcript::restore_subagents_policy(&prepared.snapshot.records).unwrap(),
            expected
        );
        crate::session::lifecycle::prepare_new_session_install_with_route(
            &agent, &live, prepared, None,
        )
        .unwrap()
        .commit(&mut agent, &live);
        assert_eq!(agent.subagents_policy(), &expected);
        let resumed = crate::session::prepare_resume_package(dir.path(), &original_id).unwrap();
        crate::session::restore::prepare_routed_resume_install(&agent, &live, resumed, None)
            .unwrap()
            .commit(&mut agent, &live);
        assert_eq!(
            agent.subagents_policy(),
            &crate::config::SubagentsConfig::default()
        );
        assert_eq!(engine_config.new_session_subagents_policy, expected);
    }

    #[test]
    fn reload_failure_preserves_engine_owned_state() {
        let old_path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-old-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let bad_path = old_path.with_file_name("letcode-engine-reload-bad.toml");
        let old_contents = r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "old"
            [providers.primary.auth]
            type = "bearer"
            credential = "old-key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"

            [providers.primary.models.old]
            [providers.primary.models.new]
            "#;
        let bad_contents = r#"
            active_provider = "primary"

            [providers.primary]
            unknown = true
            protocol = "responses"
            default_model = "new"
            [providers.primary.auth]
            type = "bearer"
            credential = "new-key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"

            [providers.primary.models.new]
            "#;
        fs::write(&old_path, old_contents).expect("write old config");
        fs::write(&bad_path, bad_contents).expect("write invalid reload config");
        let old_config = AppConfig::load_from_path(&old_path).expect("old config should load");
        let route = ModelRoute::new("primary", "old");
        let mut agent = Agent::new("old", 1, 1);
        agent.set_primary_route(route.clone());

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_default_routes =
            indexmap::IndexMap::from([(String::from("explorer"), route.clone())]);

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();
        let mut providers = old_config.providers.clone();
        let mut global_retry = old_config.global.retry.clone();
        let mut provider_api_key_hints =
            indexmap::IndexMap::from([(String::from("primary"), String::from("old hint"))]);
        let mut new_session_default_route = route.clone();
        let mut runtime_catalog = old_config.runtime_catalog.clone();
        let old_model_routes = model_routes.clone();
        let old_route_api_key_configured = route_api_key_configured.clone();
        let old_expert_default_routes = expert_default_routes.clone();
        let old_providers = providers.clone();
        let old_global_retry = global_retry.clone();
        let old_provider_api_key_hints = provider_api_key_hints.clone();
        let old_new_session_default_route = new_session_default_route.clone();
        let mut engine_config = test_engine_config(
            bad_path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        assert!(
            reload_for_test(
                &mut agent,
                &mut engine_config,
                &mut route_api_key_configured,
                &mut provider_api_key_hints,
                &event_tx,
            )
            .is_err()
        );

        assert_eq!(agent.primary_route(), Some(&route));
        assert_eq!(engine_config.model_routes, old_model_routes);
        assert_eq!(route_api_key_configured, old_route_api_key_configured);
        assert_eq!(
            engine_config.expert_default_routes,
            old_expert_default_routes
        );
        assert_eq!(
            engine_config.providers.keys().collect::<Vec<_>>(),
            old_providers.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            engine_config.providers["primary"].api_key,
            old_providers["primary"].api_key
        );
        assert_eq!(
            engine_config.providers["primary"]
                .models
                .keys()
                .collect::<Vec<_>>(),
            old_providers["primary"].models.keys().collect::<Vec<_>>()
        );
        assert_eq!(engine_config.global_retry, old_global_retry);
        assert_eq!(provider_api_key_hints, old_provider_api_key_hints);
        assert_eq!(
            engine_config.new_session_default_route,
            old_new_session_default_route
        );
        assert_eq!(
            engine_config.runtime_catalog.fingerprint(),
            old_config.runtime_catalog.fingerprint()
        );
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(old_path);
        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn config_reload_preserves_session_reasoning_effort() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-non-runtime-write-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [mcp.alpha]
            type = "local"
            command = ["alpha"]
            enabled = true

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            [providers.primary.models.primary-model.capabilities]
            reasoning = true
            [providers.primary.models.primary-model.capabilities.generation]
            reasoning = true
            [providers.primary.models.primary-model.generation]
            reasoning_effort = "high"
            "#,
        )
        .expect("write initial config");

        let config = AppConfig::load_from_path(&path).expect("initial config should load");
        let route = config.active_route();
        let provider = config
            .provider_for_route(&route)
            .expect("active provider is configured");
        let mut agent = Agent::new(route.model.clone(), 1, 1);
        agent.set_primary_route(route.clone());
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
        );
        agent.set_model_catalog(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
        );
        agent
            .set_reasoning_effort(ModelReasoningEffort::High)
            .expect("set session reasoning effort");
        agent.set_compaction_config(config.global.compaction.clone());
        agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
        agent
            .set_tool_parallelism(
                config
                    .tools
                    .parallelism
                    .iter()
                    .map(|(name, mode)| (name.clone(), *mode)),
            )
            .expect("tool parallelism");
        agent.set_retry_config(
            provider
                .retry
                .clone()
                .unwrap_or_else(|| config.global.retry.clone()),
        );

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_default_routes = crate::delegation::supported_agent_names()
            .filter_map(|name| {
                config
                    .expert_route_for(name)
                    .map(|route| (name.to_string(), route))
            })
            .collect();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();
        let mut providers = config.providers.clone();
        let mut global_retry = config.global.retry.clone();
        let mut provider_api_key_hints = config
            .providers
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    format!(
                        "Set [providers.{name}.auth].credential in {} or set {} environment variable.",
                        config.config_path.display(),
                        crate::config::provider_api_key_env_var(name)
                    ),
                )
            })
            .collect();
        let mut new_session_default_route = route;
        let mut runtime_catalog = config.runtime_catalog.clone();

        crate::config::persist_mcp_server_enabled(&path, "alpha", false)
            .expect("persist non-reloadable global setting");
        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("non-reloadable config write should preserve session settings");

        assert_eq!(
            agent.reasoning_effort(),
            Some(ModelReasoningEffort::High),
            "configuration reload must preserve the session reasoning selection"
        );
        assert!(event_rx.try_recv().is_err());
        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn reload_keeps_current_route_when_it_leaves_the_catalog() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-removed-current-route-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            [providers.primary.models.session-model]
            "#,
        )
        .expect("write initial config");

        let current_route = ModelRoute::new("primary", "session-model");
        let default_route = ModelRoute::new("primary", "primary-model");
        let mut agent = Agent::new(current_route.model.clone(), 1, 1);
        agent.set_primary_route(current_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_default_routes = indexmap::IndexMap::new();

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();

        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = current_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            "#,
        )
        .expect("remove current session model from config");

        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("catalog reload should keep the current route alive");

        assert_eq!(agent.primary_route(), Some(&current_route));
        assert_eq!(engine_config.new_session_default_route, default_route);
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true),
            "the retained current route keeps its credential state"
        );
        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::Notice(notice)
                if notice.message.contains("is no longer in the configured model catalog")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::ModelCatalogUpdated(catalog)
                if catalog.models.iter().all(|model| model.id != current_route.display_name())
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelChanged { .. }))
        );

        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("duplicate removed-route reload should be a no-op");
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn reload_keeps_current_route_when_its_provider_leaves_the_catalog() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-removed-current-provider-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "session-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"
            [providers.primary.models.session-model]
            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"
            [providers.secondary.models.secondary-model]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let current_route = ModelRoute::new("primary", "session-model");
        let next_default = ModelRoute::new("secondary", "secondary-model");
        let provider = initial_config.provider_for_route(&current_route).unwrap();
        let mut agent = Agent::new(current_route.model.clone(), 1, 1);
        agent.set_primary_route(current_route.clone());
        let mut model_routes = indexmap::IndexMap::from([
            (current_route.display_name(), current_route.clone()),
            (next_default.display_name(), next_default.clone()),
        ]);
        let mut route_api_key_configured = indexmap::IndexMap::from([
            (current_route.display_name(), true),
            (next_default.display_name(), true),
        ]);
        let mut expert_default_routes = indexmap::IndexMap::new();

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();

        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = current_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        let changed = r#"
            active_provider = "secondary"
            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"
            [providers.secondary.models.secondary-model]
        "#;
        fs::write(&path, changed).expect("remove current provider");

        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("removed provider reload keeps live route");

        assert_eq!(agent.primary_route(), Some(&current_route));
        assert_eq!(engine_config.new_session_default_route, next_default);
        assert!(
            engine_config
                .providers
                .get("primary")
                .is_some_and(|provider| {
                    provider.has_model("session-model") && !provider.api_key.is_empty()
                })
        );
        assert!(
            !engine_config
                .model_routes
                .contains_key(&current_route.display_name())
        );
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true)
        );
        while event_rx.try_recv().is_ok() {}
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("duplicate removed-provider reload should be a no-op");
        assert!(event_rx.try_recv().is_err());
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true)
        );
        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn reload_runtime_change_updates_fingerprint_and_reinstalls_current_route() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-runtime-change-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "model"
            [providers.primary.auth]
            type = "bearer"
            credential = "key"
            [providers.primary.endpoints]
            base_url = "https://old.example.invalid/v1"
            [providers.primary.models.model]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let route = initial_config.active_route();
        let provider = initial_config.active_provider().1;
        let mut agent = Agent::new(route.model.clone(), 1, 1);
        agent.set_primary_route(route.clone());
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
        );
        agent.set_model_catalog(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
        );
        agent.install_provider_usage_anchor_for_test(crate::agent::TokenUsageEstimate {
            used_tokens: 10,
            context_window_tokens: 1_000,
            input_tokens: 8,
            output_tokens: 2,
            cached_tokens: 0,
        });

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_default_routes = indexmap::IndexMap::new();

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();

        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, _event_rx) = mpsc::unbounded_channel();

        let changed = initial.replace(
            "https://old.example.invalid/v1",
            "https://new.example.invalid/v1",
        );
        fs::write(&path, changed).expect("write changed config");
        let expected = AppConfig::load_from_path(&path)
            .expect("load changed config")
            .runtime_catalog
            .fingerprint()
            .clone();

        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("runtime change reload succeeds");

        assert_eq!(engine_config.runtime_catalog.fingerprint(), &expected);
        assert_eq!(agent.primary_route(), Some(&route));
        assert!(
            agent.provider_usage_anchor_for_test().is_none(),
            "reinstalling a changed route clears provider-local usage state"
        );
        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn default_model_only_reload_updates_defaults_without_reinstalling_active_route() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-default-only-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "a"
            [providers.primary.auth]
            type = "bearer"
            credential = "key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"
            [providers.primary.models.a]
            [providers.primary.models.b]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let active_route = ModelRoute::new("primary", "a");
        let next_default = ModelRoute::new("primary", "b");
        let provider = initial_config.active_provider().1;
        let mut agent = Agent::new(active_route.model.clone(), 1, 1);
        agent.set_primary_route(active_route.clone());
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
        );
        agent.set_model_catalog(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
        );
        let usage = crate::agent::TokenUsageEstimate {
            used_tokens: 10,
            context_window_tokens: 1_000,
            input_tokens: 8,
            output_tokens: 2,
            cached_tokens: 0,
        };
        agent.install_provider_usage_anchor_for_test(usage);

        let mut model_routes = initial_config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.keys().map(move |model| {
                    let route = ModelRoute::new(provider_name, model);
                    (route.display_name(), route)
                })
            })
            .collect();
        let mut route_api_key_configured = indexmap::IndexMap::from([
            (active_route.display_name(), true),
            (next_default.display_name(), true),
        ]);
        let mut expert_default_routes = indexmap::IndexMap::new();

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();

        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = active_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            initial.replace("default_model = \"a\"", "default_model = \"b\""),
        )
        .expect("write default-only config");
        let expected = AppConfig::load_from_path(&path)
            .expect("load default-only config")
            .runtime_catalog
            .fingerprint()
            .clone();

        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("default-only reload succeeds");

        assert_eq!(engine_config.runtime_catalog.fingerprint(), &expected);
        assert_eq!(engine_config.new_session_default_route, next_default);
        assert_eq!(agent.primary_route(), Some(&active_route));
        assert_eq!(agent.provider_usage_anchor_for_test(), Some(usage));
        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelCatalogUpdated(_)))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelChanged { .. }))
        );
        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[test]
    fn reload_applies_configured_active_route_change() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-active-route-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]

            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"

            [providers.secondary.models.secondary-model]
            "#,
        )
        .expect("write initial active route config");

        let primary_route = ModelRoute::new("primary", "primary-model");
        let secondary_route = ModelRoute::new("secondary", "secondary-model");
        let mut agent = Agent::new(primary_route.model.clone(), 1, 1);
        agent.set_primary_route(primary_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_default_routes = indexmap::IndexMap::new();

        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut expert_reasoning_efforts = indexmap::IndexMap::new();

        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = primary_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "secondary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]

            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"

            [providers.secondary.models.secondary-model]
            "#,
        )
        .expect("write updated active route config");

        let mut engine_config = test_engine_config(
            path,
            model_routes,
            route_api_key_configured,
            expert_default_routes,
            expert_allowed_models,
            expert_reasoning_efforts,
            providers,
            global_retry,
            provider_api_key_hints,
            new_session_default_route,
            runtime_catalog,
        );
        let mut route_api_key_configured =
            std::mem::take(&mut engine_config.route_api_key_configured);
        let mut provider_api_key_hints = std::mem::take(&mut engine_config.provider_api_key_hints);
        reload_for_test(
            &mut agent,
            &mut engine_config,
            &mut route_api_key_configured,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("active route config should reload");

        assert_eq!(agent.primary_route(), Some(&primary_route));
        assert_eq!(engine_config.new_session_default_route, secondary_route);
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::ModelCatalogUpdated(_))
        ));
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(engine_config.mcp_config_path);
    }

    #[tokio::test]
    async fn a_recorded_background_result_is_delivered_once_and_continues_the_session() {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};

        tokio::time::timeout(Duration::from_secs(30), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let (request_tx, request_rx) = tokio::sync::oneshot::channel();
            let (stop_stub_tx, stop_stub_rx) = tokio::sync::oneshot::channel();
            let stub = tokio::spawn(async move {
                tokio::time::timeout(Duration::from_secs(20), async {
                    let (stream, _) = listener.accept().await.expect("stub accepts the request");
                    let mut stream = tokio::io::BufReader::new(stream);
                    let mut content_length = None;
                    loop {
                        let mut line = String::new();
                        let read = stream.read_line(&mut line).await.expect("stub header read");
                        assert!(read > 0, "stub request closed before headers");
                        if line == "\r\n" {
                            break;
                        }
                        if let Some((name, value)) = line.split_once(':')
                            && name.eq_ignore_ascii_case("content-length")
                        {
                            content_length = Some(value.trim().parse::<usize>().unwrap());
                        }
                    }
                    let mut body = vec![0; content_length.expect("responses request has content length")];
                    stream.read_exact(&mut body).await.expect("stub body read");
                    let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    let response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
                    stream
                        .get_mut()
                        .write_all(response.as_bytes())
                        .await
                        .expect("stub response headers");
                    request_tx.send(request).expect("the request is observed");
                    stop_stub_rx.await.expect("the stub is stopped after interruption");
                    drop(stream);
                })
                .await
                .expect("responses stub timed out");
            });

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let session_id = recorder.session_id().to_string();
            let transcript = Arc::new(StdMutex::new(recorder));

            // The subagent pool records the result before it sends the command.
            transcript
                .lock()
                .unwrap()
                .record_subagent_result_structured(
                    "run-1",
                    &session_id,
                    "run-1",
                    "child-1",
                    "explorer",
                    "completed",
                    "the child finished",
                    None,
                )
                .unwrap();

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let completion = || SessionEngineCommand::BackgroundSubagentCompleted {
                generation: None,
                parent_session_id: session_id.clone(),
                parent_tool_call_id: Some("call-1".into()),
                result: Ok(crate::subagent::SubagentRunSummary {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    agent_name: "explorer".into(),
                    status: crate::subagent::SubagentStatus::Completed,
                    failure_kind: None,
                    summary: "the child finished".into(),
                    structured_result: crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "the child finished".into(),
                        malformed: false,
                        findings: Vec::new(),
                        files_read: Vec::new(),
                        files_changed: Vec::new(),
                        commands_run: Vec::new(),
                        validation: Vec::new(),
                        blockers: Vec::new(),
                        next_steps: Vec::new(),
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    },
                }),
            };

            ingress.submit_transitional(completion()).unwrap();
            tokio::time::timeout(Duration::from_secs(10), async {
                let mut deliveries = 0;
                loop {
                    match events.recv().await.expect("engine event stream") {
                        SessionTransportEvent::BackgroundSubagentCompleted { .. } => {
                            deliveries += 1;
                        }
                        SessionTransportEvent::TurnStarted => {
                            assert_eq!(deliveries, 1, "the recorded result is delivered before execution");
                            break;
                        }
                        SessionTransportEvent::AssistantDelta(_)
                        | SessionTransportEvent::ReasoningDelta(_) => {
                            panic!("execution must start before model output");
                        }
                        SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                        SessionTransportEvent::Done | SessionTransportEvent::Interrupted => {
                            panic!("the continuation ended before execution started");
                        }
                        _ => {}
                    }
                }
            })
            .await
            .expect("the continuation starts before the first response output");

            let request = tokio::time::timeout(Duration::from_secs(10), request_rx)
                .await
                .expect("the continuation reaches the provider")
                .expect("the stub reports the request");
            assert_eq!(request["model"], "model");
            assert!(
                request["input"].to_string().contains("the child finished"),
                "the model request includes the background result"
            );

            ingress.request_interrupt().unwrap();
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match events.recv().await.expect("engine event stream") {
                        SessionTransportEvent::Interrupted => break,
                        SessionTransportEvent::TurnStarted
                        | SessionTransportEvent::BackgroundSubagentCompleted { .. } => {
                            panic!("the result must start exactly one execution");
                        }
                        SessionTransportEvent::AssistantDelta(_)
                        | SessionTransportEvent::ReasoningDelta(_) => {
                            panic!("the stub holds the first model output until interruption");
                        }
                        SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                        SessionTransportEvent::Done => panic!("the held request completed before interruption"),
                        _ => {}
                    }
                }
            })
            .await
            .expect("a continuation waiting for model output can be interrupted");
            stop_stub_tx.send(()).unwrap();
            stub.await.unwrap();

            ingress.submit_transitional(completion()).unwrap();
            let (history_tx, history_rx) = tokio::sync::oneshot::channel();
            ingress
                .submit_transitional(SessionEngineCommand::InspectHistory(history_tx))
                .unwrap();
            tokio::time::timeout(Duration::from_secs(10), history_rx)
                .await
                .expect("the duplicate completion leaves the engine idle")
                .expect("the engine reports its history");
            while let Ok(event) = events.try_recv() {
                match event {
                    SessionTransportEvent::BackgroundSubagentCompleted { .. }
                    | SessionTransportEvent::TurnStarted => {
                        panic!("the same run must not be delivered or executed twice");
                    }
                    SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                    _ => {}
                }
            }
            let records = read_records(transcript.lock().unwrap().path()).unwrap();
            assert_eq!(
                records
                    .iter()
                    .filter(|record| matches!(
                        &record.event,
                        TranscriptEvent::InternalContinuation {
                            source: crate::transcript::InternalContinuationSource::SubagentCompletion,
                            ..
                        }
                    ))
                    .count(),
                1,
                "the background result continues the session once"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("background completion flow timed out");
    }

    #[tokio::test]
    async fn a_background_result_arriving_during_an_active_turn_is_delivered_and_continues() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            // A recorded user message skips title generation, so the stub only serves turns.
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let session_id = recorder.session_id().to_string();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            let release = Arc::new(tokio::sync::Notify::new());
            agent
                .try_register_tool(HoldingTool {
                    release: Some(Arc::clone(&release)),
                })
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let mut errors = Vec::new();
            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::new("first prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            // The held tool call keeps the turn running for the rest of the test.
            let barrier = tokio::time::Instant::now() + Duration::from_secs(10);
            let mut barrier_error = None;
            loop {
                match open_tool_call_batch_recorded(&transcript) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(error) => barrier_error = Some(error.to_string()),
                }
                if tokio::time::Instant::now() >= barrier {
                    panic!(
                        "the stub tool call never reached the transcript: {barrier_error:?}; errors: {errors:?}"
                    );
                }
                while let Ok(event) = events.try_recv() {
                    if let SessionTransportEvent::Error(error) = event {
                        errors.push(observed_error(&error));
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            let structured_result = crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "the child finished".into(),
                malformed: false,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-1".into(),
                child_session_id: "child-1".into(),
                raw_excerpt: None,
            };

            // The subagent pool records the result before it sends the command.
            transcript
                .lock()
                .unwrap()
                .record_subagent_result_structured(
                    "run-1",
                    &session_id,
                    "run-1",
                    "child-1",
                    "explorer",
                    "completed",
                    "the child finished",
                    Some(structured_result.clone()),
                )
                .unwrap();

            ingress
                .submit_transitional(SessionEngineCommand::BackgroundSubagentCompleted {
                    generation: None,
                    parent_session_id: session_id.clone(),
                    parent_tool_call_id: Some("call-1".into()),
                    result: Ok(crate::subagent::SubagentRunSummary {
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        agent_name: "explorer".into(),
                        status: crate::subagent::SubagentStatus::Completed,
                        failure_kind: None,
                        summary: "the child finished".into(),
                        structured_result,
                    }),
                })
                .unwrap();

            let delivery = tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match events.recv().await {
                        Some(SessionTransportEvent::BackgroundSubagentCompleted { .. }) => {
                            return true;
                        }
                        Some(_) => {}
                        None => return false,
                    }
                }
            })
            .await
            .expect("the delivered event arrives while the turn is active");
            assert!(
                delivery,
                "a result recorded while the turn is active must still be delivered"
            );

            let mut continued = false;
            for _ in 0..100 {
                let records = read_records(transcript.lock().unwrap().path()).unwrap();
                continued = records.iter().any(|record| {
                    matches!(
                        &record.event,
                        TranscriptEvent::InternalContinuation {
                            source: crate::transcript::InternalContinuationSource::SubagentCompletion,
                            ..
                        }
                    )
                });
                if continued {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(
                continued,
                "the completion must queue a continuation turn while the turn is active"
            );

            release.notify_one();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            let mut records = Vec::new();
            loop {
                while let Ok(event) = events.try_recv() {
                    if let SessionTransportEvent::Error(error) = event {
                        errors.push(observed_error(&error));
                    }
                }
                records = read_records(transcript.lock().unwrap().path()).unwrap();
                let answered = records
                    .iter()
                    .any(|record| matches!(&record.event, TranscriptEvent::ToolCallFinished { .. }));
                let continued_turn = records.iter().any(|record| {
                    matches!(&record.event, TranscriptEvent::AssistantTurn(turn) if turn.text.as_deref() == Some(STUB_ANSWER))
                });
                if (answered && continued_turn)
                    || errors
                        .iter()
                        .any(|error| error.contains("orphan tool output"))
                {
                    break;
                }
                if tokio::time::Instant::now() >= deadline {
                    panic!("the released tool call never closed its batch: {errors:?}");
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            assert!(
                !errors
                    .iter()
                    .any(|error| error.contains("orphan tool output")),
                "{errors:?}"
            );
            let history =
                crate::transcript::transcript_projection::restore_session_history_projection(
                    &records,
                );
            crate::protocol_frames::validate_history_items_complete(&history, None)
                .expect("the recorded turn stays projectable");
            let output = history
                .iter()
                .position(|item| {
                    matches!(
                        item,
                        crate::request_builder::HistoryItem::ToolOutput { call_id, .. }
                            if call_id == "call-1"
                    )
                })
                .expect("tool output");
            let notice = history
                .iter()
                .position(|item| {
                    matches!(
                        item,
                        crate::request_builder::HistoryItem::InternalContinuation { .. }
                    )
                })
                .expect("completion notice");
            assert!(output < notice, "{history:?}");

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            stub.abort();
        })
        .await
        .expect("background completion during an active turn timed out");
    }

    #[tokio::test]
    async fn local_and_peer_inputs_wait_for_the_active_turn_and_keep_fifo_order() {
        for peer_first in [false, true] {
            check_queued_inputs(peer_first, None).await;
        }
    }

    #[tokio::test]
    async fn prompt_cancellation_targets_queued_and_active_submissions() {
        for active in [false, true] {
            check_queued_inputs(false, Some(active)).await;
        }
    }

    #[tokio::test]
    async fn local_prompt_inserts_at_the_next_tool_batch_boundary() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let release = Arc::new(tokio::sync::Notify::new());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            agent
                .try_register_tool(HoldingTool {
                    release: Some(Arc::clone(&release)),
                })
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::from("first prompt"),
                    ),
                ))
                .unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while !open_tool_call_batch_recorded(&transcript).unwrap_or(false) {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "the stub tool call never reached the transcript"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-2",
                        crate::user_content::UserMessageContent::from("second prompt"),
                    ),
                ))
                .unwrap();
            release.notify_one();

            let mut assistant_done = None;
            let mut second_user_message = None;
            let mut done = false;
            while !done {
                match events.recv().await.expect("engine events") {
                    SessionTransportEvent::AssistantDone { .. } => {
                        assistant_done.get_or_insert(());
                    }
                    SessionTransportEvent::UserMessage(message)
                        if message.submission_id == "prompt-2" =>
                    {
                        assert!(assistant_done.is_some());
                        second_user_message.get_or_insert(());
                    }
                    SessionTransportEvent::Done => done = true,
                    SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                    _ => {}
                }
            }
            assert!(assistant_done.is_some());
            assert!(second_user_message.is_some());

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            stub.abort();
        })
        .await
        .expect("boundary prompt insertion timed out");
    }

    #[tokio::test]
    async fn peer_message_inserts_at_the_next_tool_batch_boundary() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let release = Arc::new(tokio::sync::Notify::new());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            agent
                .try_register_tool(HoldingTool {
                    release: Some(Arc::clone(&release)),
                })
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            let peers = crate::peer::PeerRuntime::start(
                directory.path(),
                None,
                "sender".into(),
                None,
                |_| {},
            )
            .await
            .unwrap();
            let peer_address = loop {
                let discovered = peers.client().discover().unwrap();
                if let Some(peer) = discovered
                    .into_iter()
                    .find(|peer| peer.address.session_id != "sender")
                {
                    break peer.address;
                }
                tokio::task::yield_now().await;
            };

            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::from("first prompt"),
                    ),
                ))
                .unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while !open_tool_call_batch_recorded(&transcript).unwrap_or(false) {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "the stub tool call never reached the transcript"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            ingress
                .submit_transitional(SessionEngineCommand::PeerMessage(
                    crate::peer::PeerMessage {
                        from: peer_address.clone(),
                        to: peer_address.clone(),
                        message: "peer input".into(),
                    },
                ))
                .unwrap();
            release.notify_one();

            let mut assistant_done = false;
            let mut peer_user_message = false;
            let mut done = false;
            while !done {
                match events.recv().await.expect("engine events") {
                    SessionTransportEvent::AssistantDone { .. } => assistant_done = true,
                    SessionTransportEvent::UserMessage(message)
                        if message.content.text.contains("peer input") =>
                    {
                        assert!(assistant_done);
                        peer_user_message = true;
                    }
                    SessionTransportEvent::Done => done = true,
                    SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                    _ => {}
                }
            }
            assert!(assistant_done);
            assert!(peer_user_message);

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            assert_eq!(peers.client().discover().unwrap().len(), 1);
            stub.abort();
        })
        .await
        .expect("peer boundary prompt insertion timed out");
    }

    async fn check_queued_inputs(peer_first: bool, cancel_active: Option<bool>) {
        tokio::time::timeout(Duration::from_secs(30), async {
            let release = Arc::new(tokio::sync::Notify::new());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
submit_while_running = "queue"
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            // A recorded user message skips title generation, so the stub only serves turns.
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            agent
                .try_register_tool(HoldingTool { release: Some(release.clone()) })
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings =
                crate::session_engine_config(&config, Default::default(), String::new(), None);
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let peers = crate::peer::PeerRuntime::start(
                directory.path(), None, "sender".into(), None, |_| {},
            ).await.unwrap();
            let peer_address = loop {
                let discovered = peers.client().discover().unwrap();
                if let Some(peer) = discovered.into_iter().find(|peer| peer.address.session_id != "sender") {
                    break peer.address;
                }
                tokio::task::yield_now().await;
            };
            if let Some(cancel_active) = cancel_active {
                let prompt = |id: &str| SessionEngineCommand::Prompt(
                    crate::user_content::UserMessageSubmission::new(id, id.into()),
                );
                let peer = || SessionEngineCommand::PeerMessage(crate::peer::PeerMessage {
                    from: peer_address.clone(),
                    to: peer_address.clone(),
                    message: "peer input".into(),
                });
                ingress.submit_transitional(if cancel_active { prompt("target") } else { peer() }).unwrap();
                loop {
                    if open_tool_call_batch_recorded(&transcript).unwrap_or(false) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                if cancel_active {
                    ingress.submit_transitional(peer()).unwrap();
                } else {
                    ingress.submit_transitional(prompt("target")).unwrap();
                }
                ingress.cancel_prompt("target").unwrap();
                let mut accepted = Vec::new();
                loop {
                    match events.recv().await.unwrap() {
                        SessionTransportEvent::QueuedPromptAccepted { prompt } => accepted.push(prompt.id),
                        SessionTransportEvent::Interrupted => {
                            assert!(cancel_active);
                            break;
                        }
                        SessionTransportEvent::PromptCancelled { submission_id } => {
                            assert!(!cancel_active);
                            assert_eq!(submission_id, "target");
                            break;
                        }
                        SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                        _ => {}
                    }
                }
                assert_eq!(accepted.len(), 1);
                assert_eq!(accepted[0] == "target", cancel_active);
                if !cancel_active {
                    assert!(open_tool_call_batch_recorded(&transcript).unwrap());
                    release.notify_one();
                }
                ingress.cancel_prompt("target").unwrap();
                loop {
                    match events.recv().await.unwrap() {
                        SessionTransportEvent::Done => break,
                        SessionTransportEvent::QueuedPromptAccepted { prompt } => {
                            assert!(cancel_active);
                            assert!(prompt.text().contains("peer input"));
                        }
                        SessionTransportEvent::Interrupted | SessionTransportEvent::PromptCancelled { .. } => {
                            panic!("completed or unrelated submissions stay unchanged");
                        }
                        SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                        _ => {}
                    }
                }
                ingress.shutdown().unwrap();
                engine.join().await.unwrap();
                stub.abort();
                return;
            }
            let mut errors = Vec::new();
            let mut accepted = Vec::new();
            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::new("first prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            let barrier = tokio::time::Instant::now() + Duration::from_secs(10);
            let mut barrier_error = None;
            loop {
                match open_tool_call_batch_recorded(&transcript) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(error) => barrier_error = Some(error.to_string()),
                }
                if tokio::time::Instant::now() >= barrier {
                    panic!(
                        "the stub tool call never reached the transcript: {barrier_error:?}; errors: {errors:?}"
                    );
                }
                while let Ok(event) = events.try_recv() {
                    match event {
                        SessionTransportEvent::Error(error) => errors.push(observed_error(&error)),
                        SessionTransportEvent::QueuedPromptAccepted { prompt } => accepted.push(prompt.content.text),
                        _ => {}
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-2",
                        crate::user_content::UserMessageContent::new("second prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            let peer_input = SessionEngineCommand::PeerMessage(crate::peer::PeerMessage {
                from: crate::peer::PeerAddress {
                    instance_id: "source".into(), session_id: "sender".into(), generation: 0,
                },
                to: peer_address.clone(),
                message: "peer input".into(),
            });
            ingress.submit(SessionCommand::CancelQueuedPrompt("prompt-2".into())).unwrap();
            let queued = |id: &str| SessionEngineCommand::Prompt(
                crate::user_content::UserMessageSubmission::new(id, crate::user_content::UserMessageContent::from(id)),
            );
            let commands = if peer_first {
                [peer_input, queued("local input")]
            } else {
                [queued("local input"), peer_input]
            };
            for command in commands {
                ingress.submit_transitional(command).unwrap();
            }
            let (reply, mut barrier) = tokio::sync::oneshot::channel();
            ingress.submit_transitional(SessionEngineCommand::InspectHistory(reply)).unwrap();
            assert!(tokio::time::timeout(Duration::from_millis(30), &mut barrier).await.is_err());
            assert!(open_tool_call_batch_recorded(&transcript).unwrap());
            release.notify_one();
            barrier.await.unwrap();
            let mut done = 0;
            while done != 3 {
                match events.recv().await.unwrap() {
                    SessionTransportEvent::QueuedPromptAccepted { prompt } => accepted.push(prompt.content.text),
                    SessionTransportEvent::Done => done += 1,
                    SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                    _ => {}
                }
            }
            assert_eq!(accepted.len(), 3);
            assert_eq!(accepted[0], "first prompt");
            assert!(accepted[if peer_first { 1 } else { 2 }].contains("peer input"));
            assert_eq!(accepted[if peer_first { 2 } else { 1 }], "local input");
            assert!(errors.is_empty());
            ingress.submit(SessionCommand::NewSession).unwrap();
            loop {
                match events.recv().await.unwrap() {
                    SessionTransportEvent::SessionStarted { .. } => break,
                    SessionTransportEvent::Error(error) => panic!("{}", observed_error(&error)),
                    _ => {}
                }
            }
            ingress.submit_transitional(SessionEngineCommand::PeerMessage(crate::peer::PeerMessage {
                from: peer_address.clone(), to: peer_address, message: "old target".into(),
            })).unwrap();
            let (reply, barrier) = tokio::sync::oneshot::channel();
            ingress.submit_transitional(SessionEngineCommand::InspectHistory(reply)).unwrap();
            barrier.await.unwrap();
            while let Ok(event) = events.try_recv() {
                assert!(!matches!(event, SessionTransportEvent::QueuedPromptAccepted { .. }));
                if let SessionTransportEvent::Error(error) = event {
                    panic!("{}", observed_error(&error));
                }
            }
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            assert_eq!(peers.client().discover().unwrap().len(), 1);
            stub.abort();
        })
        .await
        .expect("queued inputs timed out");
    }
}
