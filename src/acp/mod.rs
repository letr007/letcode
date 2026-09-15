//! ACP (Agent Client Protocol) frontend.
//!
//! Serves the session engine to external ACP clients over stdio, so an editor
//! or remote-control hub can drive a letcode session without a terminal.
//!
//! Connection callbacks registered through `agent_client_protocol::Builder`
//! run inside the SDK dispatch loop, which blocks until the callback returns.
//! Awaiting a client response from inside such a callback therefore deadlocks;
//! callbacks only forward requests to [`driver`], which owns the client round
//! trips and runs as the connection's foreground future.

mod driver;
mod projection;
mod session_state;
mod slash;

use std::path::PathBuf;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, LoadSessionRequest, NewSessionRequest, PromptCapabilities, PromptRequest,
    SessionCapabilities, SessionListCapabilities, SetSessionConfigOptionRequest,
    SetSessionModeRequest,
};
use agent_client_protocol::{Agent, ConnectTo, Stdio};
use anyhow::Result;
use tokio::sync::mpsc;

use crate::session::{SessionEngine, SessionEngineProjection};

use driver::DriverRequest;
use session_state::SessionState;
pub use session_state::{ModelSummary, SessionSettings};

/// Local directories the frontend serves session lifecycle requests from.
pub struct SessionLocations {
    /// Where session transcripts live; `session/list` and `session/load`
    /// resolve session ids here.
    pub sessions_dir: PathBuf,
    /// The workspace sessions run in, reported as the working directory of a
    /// listed session.
    pub workspace_dir: PathBuf,
}

/// Serves `engine` to an ACP client over stdio until it disconnects.
///
/// The engine is started by the caller and already owns a session; that
/// session backs the first `session/new` the client sends.
pub async fn run(
    engine: SessionEngine,
    projection: SessionEngineProjection,
    settings: SessionSettings,
    locations: SessionLocations,
) -> Result<()> {
    run_over(engine, projection, settings, locations, Stdio::new()).await
}

/// Serves `engine` over `transport` until the client disconnects.
async fn run_over<T: ConnectTo<Agent> + 'static>(
    engine: SessionEngine,
    projection: SessionEngineProjection,
    settings: SessionSettings,
    locations: SessionLocations,
    transport: T,
) -> Result<()> {
    let (requests_tx, requests_rx) = mpsc::unbounded_channel();
    let builder = Agent
        .builder()
        .name("letcode")
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: InitializeRequest, responder, _connection| {
                    let version = ProtocolVersion::V1;
                    if request.protocol_version != version {
                        tracing::debug!(
                            requested = ?request.protocol_version,
                            supported = ?version,
                            "client requested another ACP protocol version"
                        );
                    }
                    // What the client can answer is settled by its handshake,
                    // so the driver learns it before the client sends anything
                    // that would need it.
                    forward(
                        &requests_tx,
                        DriverRequest::Initialized {
                            capabilities: request.client_capabilities,
                        },
                    );
                    responder.respond(
                        InitializeResponse::new(version)
                            .agent_capabilities(
                                AgentCapabilities::new()
                                    .load_session(true)
                                    .prompt_capabilities(PromptCapabilities::new().image(true))
                                    .session_capabilities(
                                        SessionCapabilities::new()
                                            .list(SessionListCapabilities::new()),
                                    ),
                            )
                            .agent_info(Implementation::new("letcode", env!("CARGO_PKG_VERSION"))),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |_request: NewSessionRequest, responder, _connection| {
                    forward(&requests_tx, DriverRequest::NewSession { responder });
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: LoadSessionRequest, responder, _connection| {
                    forward(
                        &requests_tx,
                        DriverRequest::Load {
                            session_id: request.session_id,
                            responder,
                        },
                    );
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: ListSessionsRequest, responder, _connection| {
                    forward(
                        &requests_tx,
                        DriverRequest::ListSessions {
                            cwd: request.cwd,
                            cursor: request.cursor,
                            responder,
                        },
                    );
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: PromptRequest, responder, _connection| {
                    match driver::prompt_content(&request.prompt) {
                        Ok(content) => forward(
                            &requests_tx,
                            DriverRequest::Prompt {
                                session_id: request.session_id,
                                content,
                                responder,
                            },
                        ),
                        Err(error) => {
                            let _ = responder.respond_with_error(error);
                        }
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: SetSessionModeRequest, responder, _connection| {
                    forward(
                        &requests_tx,
                        DriverRequest::SetMode {
                            session_id: request.session_id,
                            mode_id: request.mode_id,
                            responder,
                        },
                    );
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let requests_tx = requests_tx.clone();
                async move |request: SetSessionConfigOptionRequest, responder, _connection| {
                    forward(
                        &requests_tx,
                        DriverRequest::SetConfigOption {
                            session_id: request.session_id,
                            config_id: request.config_id,
                            value: request.value,
                            responder,
                        },
                    );
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let requests_tx = requests_tx.clone();
                async move |notification: CancelNotification, _connection| {
                    forward(
                        &requests_tx,
                        DriverRequest::Cancel {
                            session_id: notification.session_id,
                        },
                    );
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        );

    let state = SessionState::new(
        &projection.permission_mode_label,
        &projection.model_id,
        &settings,
    );
    let initial_session_id = projection.session_id;
    builder
        .connect_with(transport, async move |connection| {
            driver::run(
                engine,
                initial_session_id,
                state,
                locations,
                connection,
                requests_rx,
            )
            .await
        })
        .await
        .map_err(|error| anyhow::anyhow!(error).context("ACP connection terminated with an error"))
}

/// Hands a request to the driver, or fails it when the driver is gone.
///
/// A request that never reaches the driver is answered instead of hanging.
fn forward(requests_tx: &mpsc::UnboundedSender<DriverRequest>, request: DriverRequest) {
    if let Err(error) = requests_tx.send(request) {
        error.0.fail("letcode session driver is not running");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use agent_client_protocol::schema::v1::{
        AudioContent, ContentBlock, ContentChunk, InitializeRequest, NewSessionRequest,
        PromptRequest, PromptResponse, SessionConfigOption, SessionConfigOptionValue, SessionId,
        SessionModeId, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
        SetSessionModeRequest, StopReason, TextContent,
    };
    use agent_client_protocol::{AcpAgent, AcpAgentConfig, Channel, Client, ConnectionTo};
    use tokio::sync::mpsc;

    use crate::agent::{Agent, ConfiguredPrimaryRouteFactory, PrimaryRouteFactory as _};
    use crate::config::{AppConfig, ModelRoute};
    use crate::request_builder::ModelReasoningEffort;
    use crate::session::SessionEngine;
    use crate::transcript::{TranscriptRecorder, read_records};

    /// A provider that is never reached: the handshake only exercises session
    /// setup and settings, which the engine resolves without a model request.
    const TEST_CONFIG: &str = r#"
active_provider = "test"

[permissions]
mode = "auto"

[providers.test]
protocol = "responses"
default_model = "model"

[providers.test.auth]
type = "bearer"
credential = "test-key"

[providers.test.endpoints]
base_url = "http://127.0.0.1:1"

[providers.test.models.model]

[providers.test.models.model.capabilities]
reasoning = true
generation = { reasoning = true }

[providers.test.models.model.generation]
reasoning_effort = "low"
reasoning_efforts = ["low", "high"]

[providers.test.models.other]
"#;

    fn config_value(options: &[SessionConfigOption], id: &str) -> Option<String> {
        let option = options.iter().find(|option| option.id.0.as_ref() == id)?;
        let agent_client_protocol::schema::v1::SessionConfigKind::Select(select) = &option.kind
        else {
            panic!("expected a select option for {id}");
        };
        Some(select.current_value.0.to_string())
    }

    /// The text of a replayed message chunk.
    fn chunk_text(chunk: &ContentChunk) -> String {
        match &chunk.content {
            ContentBlock::Text(text) => text.text.clone(),
            other => panic!("expected replayed text content, got {other:?}"),
        }
    }

    /// Sends prompt text as a prompt for `session_id`.
    async fn send_prompt(
        connection: &ConnectionTo<agent_client_protocol::Agent>,
        session_id: &SessionId,
        text: &str,
    ) -> Result<PromptResponse, agent_client_protocol::Error> {
        connection
            .send_request(PromptRequest::new(
                session_id.clone(),
                vec![ContentBlock::Text(TextContent::new(text))],
            ))
            .block_task()
            .await
    }

    /// Waits for the notification a test asserts on, skipping the ones it does
    /// not.
    async fn await_notification(
        updates: &mut mpsc::UnboundedReceiver<SessionNotification>,
        matches: impl Fn(&SessionNotification) -> bool,
    ) -> SessionNotification {
        loop {
            let notification = tokio::time::timeout(Duration::from_secs(10), updates.recv())
                .await
                .expect("timed out waiting for a session update")
                .expect("session updates closed");
            if matches(&notification) {
                return notification;
            }
        }
    }

    /// Waits for the update a test asserts on, skipping the ones it does not.
    async fn await_update(
        updates: &mut mpsc::UnboundedReceiver<SessionNotification>,
        matches: impl Fn(&SessionUpdate) -> bool,
    ) -> SessionUpdate {
        await_notification(updates, |notification| matches(&notification.update))
            .await
            .update
    }

    /// Starts an engine whose only session is the one the ACP frontend adopts,
    /// mirroring the settings `main` installs before serving ACP.
    fn start_engine(
        config: &AppConfig,
    ) -> (
        SessionEngine,
        SessionEngineProjection,
        Option<ModelReasoningEffort>,
        Arc<Mutex<TranscriptRecorder>>,
    ) {
        let route = ModelRoute::new("test", "model");
        let factory = Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
            config.providers.clone(),
            config.global.retry.clone(),
            config.runtime_catalog.clone(),
        ));
        let mut agent = Agent::new("model", 1, 1);
        agent.set_permission_mode(config.permissions.mode);
        agent.set_model_catalog(
            config.providers["test"]
                .models
                .iter()
                .map(|(model_id, model)| (model_id.clone(), model.request_metadata()))
                .collect::<HashMap<_, _>>(),
        );
        agent.apply_prepared_route(factory.prepare_route(route.clone()).expect("prepare route"));
        agent.set_primary_route_factory(factory);
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&config.global.sessions_dir).expect("create transcript"),
        ));
        transcript
            .lock()
            .expect("transcript lock")
            .record_session_started(route.display_name())
            .expect("record session start");
        crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

        let reasoning_effort = agent.reasoning_effort();
        let engine_config = crate::session_engine_config(config, Default::default(), String::new());
        let (engine, projection) = SessionEngine::start(
            agent,
            Arc::clone(&transcript),
            "Model".to_string(),
            engine_config,
        )
        .expect("start engine");
        (engine, projection, reasoning_effort, transcript)
    }

    /// The directories the frontend serves session lifecycle requests from.
    fn locations(config: &AppConfig) -> SessionLocations {
        SessionLocations {
            sessions_dir: config.global.sessions_dir.clone(),
            workspace_dir: std::env::current_dir().expect("current directory"),
        }
    }

    /// Writes a session a client can load: a finished turn, a title, the route
    /// it ran with, and the permission mode its transcript records.
    fn write_resumable_session(
        sessions_dir: &std::path::Path,
        model: &str,
        mode: Option<&str>,
    ) -> String {
        let mut recorder =
            TranscriptRecorder::create(sessions_dir).expect("create fixture transcript");
        recorder
            .record_session_started(model)
            .expect("record fixture session start");
        recorder
            .record_session_title("Fixture session")
            .expect("record fixture title");
        recorder
            .record_user_message("fixture question")
            .expect("record fixture prompt");
        recorder
            .record_assistant_message("fixture answer")
            .expect("record fixture reply");
        if let Some(mode) = mode {
            recorder
                .record_permission_mode_changed("default", mode)
                .expect("record fixture mode");
        }
        recorder.session_id().to_string()
    }

    fn session_settings(reasoning_effort: Option<ModelReasoningEffort>) -> SessionSettings {
        SessionSettings::new(
            vec![
                ModelSummary::new(
                    "test/model",
                    "Model",
                    vec![ModelReasoningEffort::Low, ModelReasoningEffort::High],
                ),
                ModelSummary::new("test/other", "Other", Vec::new()),
            ],
            reasoning_effort,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_applies_session_mode_and_config_options() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, transcript) = start_engine(&config);
            let session_id = SessionId::new(projection.session_id.clone());

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let live_transcript = Arc::clone(&transcript);
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    let initialized = connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    assert!(
                        initialized
                            .agent_capabilities
                            .prompt_capabilities
                            .image,
                        "the agent advertises image prompt content"
                    );

                    let session = connection
                        .send_request(NewSessionRequest::new(
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(session.session_id.0.to_string(), session_id.0.to_string());
                    let modes = session.modes.expect("session modes");
                    assert_eq!(modes.current_mode_id.0.to_string(), "auto");
                    assert_eq!(modes.available_modes.len(), 4);
                    let options = session.config_options.expect("config options");
                    assert_eq!(config_value(&options, "mode").as_deref(), Some("auto"));
                    assert_eq!(
                        config_value(&options, "model").as_deref(),
                        Some("test/model")
                    );
                    assert_eq!(
                        config_value(&options, "reasoning_effort").as_deref(),
                        Some("low")
                    );

                    let error = connection
                        .send_request(PromptRequest::new(
                            session_id.clone(),
                            vec![ContentBlock::Audio(AudioContent::new("bGV0", "audio/wav"))],
                        ))
                        .block_task()
                        .await
                        .expect_err("audio prompt content is unsupported");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);

                    let error = connection
                        .send_request(SetSessionModeRequest::new(
                            session_id.clone(),
                            SessionModeId::new("plan"),
                        ))
                        .block_task()
                        .await
                        .expect_err("an unknown mode is rejected");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);

                    let error = connection
                        .send_request(SetSessionConfigOptionRequest::new(
                            session_id.clone(),
                            "nonsense",
                            SessionConfigOptionValue::value_id("value"),
                        ))
                        .block_task()
                        .await
                        .expect_err("an unknown config option is rejected");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);

                    connection
                        .send_request(SetSessionModeRequest::new(
                            session_id.clone(),
                            SessionModeId::new("yolo"),
                        ))
                        .block_task()
                        .await?;
                    let update = await_update(&mut updates, |update| {
                        matches!(update, SessionUpdate::CurrentModeUpdate(_))
                    })
                    .await;
                    let SessionUpdate::CurrentModeUpdate(mode) = update else {
                        unreachable!("the update was filtered by kind")
                    };
                    assert_eq!(mode.current_mode_id.0.to_string(), "yolo");
                    let update = await_update(&mut updates, |update| {
                        matches!(
                            update,
                            SessionUpdate::ConfigOptionUpdate(options)
                                if config_value(&options.config_options, "mode").as_deref()
                                    == Some("yolo")
                        )
                    })
                    .await;
                    let SessionUpdate::ConfigOptionUpdate(options) = update else {
                        unreachable!("the update was filtered by kind")
                    };
                    assert_eq!(
                        config_value(&options.config_options, "model").as_deref(),
                        Some("test/model"),
                        "a mode change reports the whole configuration state"
                    );

                    let response = connection
                        .send_request(SetSessionConfigOptionRequest::new(
                            session_id.clone(),
                            "reasoning_effort",
                            SessionConfigOptionValue::value_id("high"),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(
                        config_value(&response.config_options, "reasoning_effort").as_deref(),
                        Some("high")
                    );
                    await_update(&mut updates, |update| {
                        matches!(
                            update,
                            SessionUpdate::ConfigOptionUpdate(options)
                                if config_value(&options.config_options, "reasoning_effort")
                                    .as_deref()
                                    == Some("high")
                        )
                    })
                    .await;

                    let response = connection
                        .send_request(SetSessionConfigOptionRequest::new(
                            session_id.clone(),
                            "model",
                            SessionConfigOptionValue::value_id("test/other"),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(
                        config_value(&response.config_options, "model").as_deref(),
                        Some("test/other")
                    );
                    assert_eq!(
                        config_value(&response.config_options, "mode").as_deref(),
                        Some("yolo"),
                        "the response reports the whole configuration state"
                    );
                    assert_eq!(
                        config_value(&response.config_options, "reasoning_effort"),
                        None,
                        "the new route selects no reasoning level"
                    );
                    await_update(&mut updates, |update| {
                        matches!(
                            update,
                            SessionUpdate::ConfigOptionUpdate(options)
                                if config_value(&options.config_options, "model").as_deref()
                                    == Some("test/other")
                        )
                    })
                    .await;

                    // The engine records what it applies, so the transcript
                    // shows the settings reached the session itself.
                    let records = read_records(
                        live_transcript.lock().expect("transcript lock").path(),
                    )
                    .expect("read transcript");
                    assert_eq!(
                        crate::transcript::restore_latest_permission_mode(&records).as_deref(),
                        Some("yolo")
                    );
                    assert_eq!(
                        crate::transcript::restore_latest_model(&records).as_deref(),
                        Some("test/other")
                    );
                    assert_eq!(
                        crate::transcript::restore_latest_reasoning_effort(&records, "test/model"),
                        Some(ModelReasoningEffort::High)
                    );
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client handshake failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_dispatches_the_slash_commands_it_advertises() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, transcript) = start_engine(&config);
            let session_id = SessionId::new(projection.session_id.clone());
            let fixture_session_id =
                write_resumable_session(&config.global.sessions_dir, "test/model", None);

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let live_transcript = Arc::clone(&transcript);
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(NewSessionRequest::new(
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;

                    // The client learns which commands the session accepts before
                    // it sends one.
                    let update = await_update(&mut updates, |update| {
                        matches!(update, SessionUpdate::AvailableCommandsUpdate(_))
                    })
                    .await;
                    let SessionUpdate::AvailableCommandsUpdate(commands) = update else {
                        unreachable!("the update was filtered by kind")
                    };
                    assert_eq!(
                        commands
                            .available_commands
                            .iter()
                            .map(|command| command.name.as_str())
                            .collect::<Vec<_>>(),
                        vec![
                            "permission", "model", "reasoning", "compact", "fast", "new",
                            "resume", "undo", "redo",
                        ]
                    );
                    assert!(
                        commands
                            .available_commands
                            .iter()
                            .all(|command| !command.description.is_empty()),
                        "an advertised command describes itself"
                    );
                    assert!(
                        commands
                            .available_commands
                            .iter()
                            .all(|command| command.input.is_some()
                                == matches!(
                                    command.name.as_str(),
                                    "permission" | "model" | "reasoning" | "resume"
                                )),
                        "an advertised command names the value it takes, and only a command taking one does"
                    );

                    // A command is a prompt: it is answered once the engine reports
                    // the value it applied.
                    for (prompt, option, applied) in [
                        ("/permission yolo", "mode", "yolo"),
                        ("/reasoning high", "reasoning_effort", "high"),
                        ("/model test/other", "model", "test/other"),
                    ] {
                        let response = connection
                            .send_request(PromptRequest::new(
                                session_id.clone(),
                                vec![ContentBlock::Text(TextContent::new(prompt))],
                            ))
                            .block_task()
                            .await?;
                        assert_eq!(
                            response.stop_reason,
                            StopReason::EndTurn,
                            "{prompt} answered with the engine's report"
                        );
                        await_update(&mut updates, |update| {
                            matches!(
                                update,
                                SessionUpdate::ConfigOptionUpdate(options)
                                    if config_value(&options.config_options, option).as_deref()
                                        == Some(applied)
                            )
                        })
                        .await;
                    }

                    // The engine recorded what it applied, so the settings reached
                    // the session rather than this frontend's own bookkeeping.
                    let records = read_records(
                        live_transcript.lock().expect("transcript lock").path(),
                    )
                    .expect("read transcript");
                    assert_eq!(
                        crate::transcript::restore_latest_permission_mode(&records).as_deref(),
                        Some("yolo")
                    );
                    assert_eq!(
                        crate::transcript::restore_latest_model(&records).as_deref(),
                        Some("test/other")
                    );
                    assert_eq!(
                        crate::transcript::restore_latest_reasoning_effort(&records, "test/model"),
                        Some(ModelReasoningEffort::High)
                    );

                    // A command the engine cannot be asked to apply is reported as
                    // such instead of being answered as applied.
                    let error = send_prompt(&connection, &session_id, "/permission plan")
                        .await
                        .expect_err("a command without a usable value is rejected");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);
                    assert_eq!(error.message, "Usage: /permission <safe|default|auto|yolo>");

                    // A compaction the engine commits nothing for reports why it
                    // could not compact instead of starting a turn.
                    let error = send_prompt(&connection, &session_id, "/compact")
                        .await
                        .expect_err("a session with nothing to compact reports why");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InternalError);
                    assert!(
                        error
                            .message
                            .starts_with("letcode could not compact the session context"),
                        "{}",
                        error.message
                    );

                    // A command the engine refuses with a notice is answered with
                    // that notice rather than waiting for a report that will not
                    // arrive.
                    let error = send_prompt(&connection, &session_id, "/fast")
                        .await
                        .expect_err("a session without fast mode refuses the toggle");
                    assert_eq!(error.message, "Fast mode unavailable");
                    for (prompt, message) in [
                        ("/undo", "no session history entry to undo"),
                        ("/redo", "no history entry available to redo"),
                    ] {
                        let error = send_prompt(&connection, &session_id, prompt)
                            .await
                            .expect_err("history navigation without history is refused");
                        assert_eq!(error.message, message, "{prompt}");
                    }

                    // A command this frontend does not dispatch is answered as
                    // such instead of reaching the model as prompt content.
                    let error = send_prompt(&connection, &session_id, "/theme dark")
                        .await
                        .expect_err("a command ACP does not dispatch is refused");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);
                    assert_eq!(error.message, "/theme is not available over ACP");

                    // `/resume` installs the session it names, and the client
                    // addresses that session from then on.
                    let response = send_prompt(
                        &connection,
                        &session_id,
                        &format!("/resume {fixture_session_id}"),
                    )
                    .await
                    .expect("/resume answers from the session the engine installed");
                    assert_eq!(response.stop_reason, StopReason::EndTurn);
                    let resumed = SessionId::new(fixture_session_id);

                    // `/undo` and `/redo` are answered by the same report, because
                    // moving through history installs a session rather than
                    // continuing the one it moved away from.
                    for prompt in ["/undo", "/redo"] {
                        let response = send_prompt(&connection, &resumed, prompt)
                            .await
                            .unwrap_or_else(|error| panic!("{prompt} failed: {error:?}"));
                        assert_eq!(response.stop_reason, StopReason::EndTurn, "{prompt}");
                    }

                    // `/new` is answered by the session the engine starts.
                    let response = send_prompt(&connection, &resumed, "/new")
                        .await
                        .expect("/new answers from the session the engine started");
                    assert_eq!(response.stop_reason, StopReason::EndTurn);
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client slash command failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP slash command handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_loads_a_session_replaying_its_history_and_mode() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/model", Some("yolo"));
            let unmodified =
                write_resumable_session(&config.global.sessions_dir, "test/model", None);

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    let initialized = connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    assert!(
                        initialized.agent_capabilities.load_session,
                        "the agent advertises session/load"
                    );
                    assert!(
                        initialized.agent_capabilities.session_capabilities.list.is_some(),
                        "the agent advertises session/list"
                    );

                    let loaded = connection
                        .send_request(LoadSessionRequest::new(
                            SessionId::new(resumable.clone()),
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;
                    let modes = loaded.modes.expect("the load reports the session modes");
                    assert_eq!(modes.current_mode_id.0.to_string(), "yolo");
                    let options = loaded
                        .config_options
                        .expect("the load reports the configuration state");
                    assert_eq!(config_value(&options, "mode").as_deref(), Some("yolo"));

                    // The engine reports the mode it restored before the resume
                    // report, so a client knows how the session runs before it
                    // replays what happened in it.
                    let update = await_update(&mut updates, |update| {
                        matches!(update, SessionUpdate::CurrentModeUpdate(_))
                    })
                    .await;
                    let SessionUpdate::CurrentModeUpdate(mode) = update else {
                        unreachable!("the update was filtered by kind")
                    };
                    assert_eq!(mode.current_mode_id.0.to_string(), "yolo");

                    // The client sees the conversation that happened before
                    // the load.
                    await_update(&mut updates, |update| {
                        matches!(
                            update,
                            SessionUpdate::UserMessageChunk(chunk)
                                if chunk_text(chunk) == "fixture question"
                        )
                    })
                    .await;
                    await_update(&mut updates, |update| {
                        matches!(
                            update,
                            SessionUpdate::AgentMessageChunk(chunk)
                                if chunk_text(chunk) == "fixture answer"
                        )
                    })
                    .await;

                    // A session that never changed its mode keeps the mode the
                    // engine already runs with, so the load needs no report.
                    let loaded = connection
                        .send_request(LoadSessionRequest::new(
                            SessionId::new(unmodified.clone()),
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;
                    let modes = loaded.modes.expect("the load reports the session modes");
                    assert_eq!(modes.current_mode_id.0.to_string(), "yolo");
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client load failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP load handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_lists_resumable_sessions() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/model", Some("yolo"));
            let workspace = std::env::current_dir().expect("current directory");

            let (agent_transport, client_transport) = Channel::duplex();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;

                    let listed = connection
                        .send_request(ListSessionsRequest::new())
                        .block_task()
                        .await?;
                    let session = listed
                        .sessions
                        .iter()
                        .find(|session| session.session_id.0.as_ref() == resumable)
                        .expect("the resumable session is listed");
                    assert_eq!(session.title.as_deref(), Some("Fixture session"));
                    assert_eq!(session.cwd, workspace);
                    assert!(
                        session.updated_at.is_some(),
                        "a listed session reports when it was last active"
                    );

                    let filtered = connection
                        .send_request(ListSessionsRequest::new().cwd(workspace.clone()))
                        .block_task()
                        .await?;
                    assert!(
                        filtered
                            .sessions
                            .iter()
                            .any(|session| session.session_id.0.as_ref() == resumable),
                        "the workspace filter keeps the sessions of this workspace"
                    );

                    let elsewhere = connection
                        .send_request(
                            ListSessionsRequest::new().cwd(directory.path().to_path_buf()),
                        )
                        .block_task()
                        .await?;
                    assert!(
                        elsewhere.sessions.is_empty(),
                        "sessions belong to the workspace the agent serves"
                    );

                    let error = connection
                        .send_request(ListSessionsRequest::new().cursor("page-2"))
                        .block_task()
                        .await
                        .expect_err("a cursor no listing issued is rejected");
                    assert_eq!(error.code, agent_client_protocol::ErrorCode::InvalidParams);
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client listing failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP listing handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_reports_a_restored_mode_to_the_session_it_restored() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let launched = SessionId::new(projection.session_id.clone());
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/model", Some("yolo"));

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(
                            SessionId::new(resumable.clone()),
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;

                    // The loaded session is the one that runs with the mode it
                    // restored, so the report naming it is the session a client
                    // addresses from now on.
                    let notification = await_notification(&mut updates, |notification| {
                        matches!(notification.update, SessionUpdate::CurrentModeUpdate(_))
                    })
                    .await;
                    assert_eq!(notification.session_id.0.to_string(), resumable);
                    assert_ne!(notification.session_id, launched);
                    let SessionUpdate::CurrentModeUpdate(mode) = notification.update else {
                        unreachable!("the notification was filtered by update kind")
                    };
                    assert_eq!(mode.current_mode_id.0.to_string(), "yolo");
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client load failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP restored mode handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_reports_a_resumed_mode_to_the_session_it_resumed() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let launched = SessionId::new(projection.session_id.clone());
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/model", Some("yolo"));

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(NewSessionRequest::new(
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;

                    // `/resume` switches the session the engine serves, and the
                    // mode the resumed session restored belongs to that session
                    // rather than to the one it replaced.
                    let response =
                        send_prompt(&connection, &launched, &format!("/resume {resumable}"))
                            .await?;
                    assert_eq!(response.stop_reason, StopReason::EndTurn);

                    let notification = await_notification(&mut updates, |notification| {
                        matches!(notification.update, SessionUpdate::CurrentModeUpdate(_))
                    })
                    .await;
                    assert_eq!(notification.session_id.0.to_string(), resumable);
                    assert_ne!(notification.session_id, launched);
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client resume failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP resumed mode handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_reports_the_route_a_resumed_session_runs_with() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let launched = SessionId::new(projection.session_id.clone());
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/other", None);

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(NewSessionRequest::new(
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;

                    let response =
                        send_prompt(&connection, &launched, &format!("/resume {resumable}"))
                            .await?;
                    assert_eq!(response.stop_reason, StopReason::EndTurn);

                    // The route a session ran with is what a resume restores with
                    // the transcript, so the client is told which one it runs with
                    // and reads that route back from the session it addresses.
                    let notification = await_notification(&mut updates, |notification| {
                        matches!(
                            &notification.update,
                            SessionUpdate::ConfigOptionUpdate(options)
                                if config_value(&options.config_options, "model").as_deref()
                                    == Some("test/other")
                        )
                    })
                    .await;
                    assert_eq!(notification.session_id.0.to_string(), resumable);

                    let response = connection
                        .send_request(SetSessionConfigOptionRequest::new(
                            SessionId::new(resumable.clone()),
                            "mode",
                            SessionConfigOptionValue::value_id("safe"),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(
                        config_value(&response.config_options, "model").as_deref(),
                        Some("test/other"),
                        "the resumed route is the one the session runs with"
                    );
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client resume failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP resumed route handshake timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handshake_creates_a_session_after_a_load() {
        tokio::time::timeout(Duration::from_secs(60), async {
            let directory = tempfile::tempdir().expect("temp dir");
            let config_path = directory.path().join("letcode.toml");
            std::fs::write(&config_path, TEST_CONFIG).expect("write config");
            let config = AppConfig::load_from_path(&config_path).expect("load config");
            let (engine, projection, reasoning_effort, _transcript) = start_engine(&config);
            let launched = SessionId::new(projection.session_id.clone());
            let resumable =
                write_resumable_session(&config.global.sessions_dir, "test/model", None);

            let (agent_transport, client_transport) = Channel::duplex();
            let (updates_tx, mut updates) = mpsc::unbounded_channel();
            let agent = run_over(
                engine,
                projection,
                session_settings(reasoning_effort),
                locations(&config),
                agent_transport,
            );

            let client = Client
                .builder()
                .on_receive_notification(
                    async move |notification: SessionNotification,
                                _connection: ConnectionTo<agent_client_protocol::Agent>| {
                        updates_tx
                            .send(notification)
                            .map_err(agent_client_protocol::Error::into_internal_error)
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, async move |connection| {
                    connection
                        .send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;
                    connection
                        .send_request(LoadSessionRequest::new(
                            SessionId::new(resumable.clone()),
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;
                    let replayed = await_notification(&mut updates, |notification| {
                        matches!(
                            &notification.update,
                            SessionUpdate::UserMessageChunk(chunk)
                                if chunk_text(chunk) == "fixture question"
                        )
                    })
                    .await;
                    assert_eq!(
                        replayed.session_id.0.to_string(),
                        resumable,
                        "the load installs the session it replays"
                    );

                    // The session `session/new` used to serve belongs to the agent
                    // the load displaced, so asking for a new session creates one
                    // instead of handing the loaded session out again.
                    let created = connection
                        .send_request(NewSessionRequest::new(
                            std::env::current_dir().expect("current directory"),
                        ))
                        .block_task()
                        .await?;
                    assert_ne!(created.session_id.0.to_string(), resumable);
                    assert_ne!(created.session_id, launched);
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            client_result.expect("ACP client load failed");
            agent_result.expect("ACP agent connection failed");
        })
        .await
        .expect("ACP new session after load handshake timed out");
    }
}
