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
#[path = "../../tests/internal/acp/integration_tests.rs"]
mod integration_tests;
