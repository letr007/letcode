//! Drives a letcode session engine from ACP client requests.
//!
//! The driver owns the engine command ingress, the transport event stream, and
//! every round trip to the client. It runs as the connection's foreground
//! future, which is the only place where awaiting a client response is safe:
//! the SDK dispatch loop that runs the `on_receive_request` callbacks is
//! blocked until those callbacks return.

use std::path::{Path, PathBuf};

use agent_client_protocol::schema::v1::{
    ClientCapabilities, ContentBlock, CreateElicitationRequest, CreateElicitationResponse,
    ElicitationAction, ElicitationCapabilities, ElicitationContentValue, ElicitationFormMode,
    ElicitationPropertySchema, ElicitationSchema, ElicitationSessionScope, EnumOption,
    ImageContent, ListSessionsResponse, LoadSessionResponse, MultiSelectPropertySchema,
    NewSessionResponse, PermissionOption, PermissionOptionId, PermissionOptionKind, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, SelectedPermissionOutcome, SessionConfigId,
    SessionConfigOptionValue, SessionId, SessionInfo, SessionModeId, SessionNotification,
    SessionUpdate, SetSessionConfigOptionResponse, SetSessionModeResponse, StopReason,
    StringPropertySchema, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo, Error, ErrorCode, Responder};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use tokio::sync::mpsc;

use crate::agent::{ConversationMessage, ConversationRole};
use crate::permission::PermissionMode;
use crate::session::archive::merged_session_summaries;
use crate::session::{
    ErrorEvent, PermissionRequestEvent, RunnerPermissionRequest, RunnerQuestionRequest,
    SessionCommand, SessionEngine, SessionEngineIngress, SessionTransportEvent,
};
use crate::tool::{QuestionRequest, QuestionResponse, QuestionSpec};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessageSubmission};

use super::SessionLocations;
use super::projection::{AcpUpdateProjection, message_chunk};
use super::session_state::{
    CONFIG_MODEL, CONFIG_REASONING_EFFORT, CONFIG_SESSION_MODE, SessionState,
    parse_reasoning_effort,
};
use super::slash::{self, SlashRequest};

/// Permission option identifiers exchanged with the client. They are sent in
/// `session/request_permission` and echoed back in the client's outcome.
const PERMISSION_ALLOW_ONCE: &str = "allow_once";
const PERMISSION_ALLOW_ALWAYS: &str = "allow_always";
const PERMISSION_REJECT: &str = "reject_once";

/// The reason a question carries when the client advertises no form to ask it
/// through. The engine reports it to the model as the tool's outcome.
const QUESTIONS_UNSUPPORTED: &str = "the ACP client does not support form elicitation";

/// A client request forwarded from a connection callback.
///
/// Callbacks forward rather than await so the dispatch loop keeps running while
/// the driver performs the round trips the request needs.
pub(super) enum DriverRequest {
    NewSession {
        responder: Responder<NewSessionResponse>,
    },
    Load {
        session_id: SessionId,
        responder: Responder<LoadSessionResponse>,
    },
    ListSessions {
        cwd: Option<PathBuf>,
        cursor: Option<String>,
        responder: Responder<ListSessionsResponse>,
    },
    Prompt {
        session_id: SessionId,
        content: UserMessageContent,
        responder: Responder<PromptResponse>,
    },
    SetMode {
        session_id: SessionId,
        mode_id: SessionModeId,
        responder: Responder<SetSessionModeResponse>,
    },
    SetConfigOption {
        session_id: SessionId,
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
        responder: Responder<SetSessionConfigOptionResponse>,
    },
    Cancel {
        session_id: SessionId,
    },
    /// The capabilities the client advertised in its handshake. They decide
    /// which of the engine's questions this frontend can ask it.
    Initialized {
        capabilities: ClientCapabilities,
    },
}

impl DriverRequest {
    /// Answers a request that never reached the driver.
    pub(super) fn fail(self, message: &str) {
        let error = Error::new(ErrorCode::InternalError.into(), message);
        match self {
            Self::NewSession { responder } => {
                let _ = responder.respond_with_error(error);
            }
            Self::Load { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            Self::ListSessions { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            Self::Prompt { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            Self::SetMode { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            Self::SetConfigOption { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            Self::Cancel { .. } => {}
            Self::Initialized { .. } => {}
        }
    }
}

/// Serves the engine to `connection` until the client disconnects.
pub(super) async fn run(
    mut engine: SessionEngine,
    initial_session_id: String,
    state: SessionState,
    locations: SessionLocations,
    connection: ConnectionTo<Client>,
    mut requests: mpsc::UnboundedReceiver<DriverRequest>,
) -> Result<(), Error> {
    let ingress = engine.take_ingress();
    let mut events = engine.take_event_egress().into_receiver();
    let mut driver = Driver::new(initial_session_id, state, locations);
    let outcome = driver
        .drive(&connection, &ingress, &mut events, &mut requests)
        .await;

    let _ = ingress.shutdown();
    match engine.join().await {
        Ok(()) => outcome,
        Err(error) => {
            tracing::warn!(%error, "ACP session engine shutdown failed");
            outcome
        }
    }
}

/// Session state the frontend tracks across ACP requests.
struct Driver {
    /// Whether the client advertised form elicitation, the mode a question is
    /// asked through.
    elicitation_forms: bool,
    /// Engine session currently installed in the engine. ACP clients hold this
    /// id as their session id.
    active_session: String,
    /// An ACP session has been handed to the client, so the pre-created engine
    /// session is no longer available for a later `session/new`.
    session_issued: bool,
    projection: AcpUpdateProjection,
    /// Mode and configuration state mirrored from the engine.
    state: SessionState,
    turn: Option<ActiveTurn>,
    pending_new_session: Option<Responder<NewSessionResponse>>,
    pending_resume: Option<PendingResume>,
    /// The command a client sent, and the engine report that answers it.
    pending_command: Option<PendingCommand>,
    /// A mode the engine announced while installing a session. The session that
    /// runs with it is not installed yet, so the report waits for it.
    pending_mode_report: bool,
    /// Where the frontend resolves the sessions a client can load.
    sessions_dir: PathBuf,
    /// The workspace sessions run in, reported as the working directory of a
    /// listed session.
    workspace_dir: PathBuf,
}

struct ActiveTurn {
    responder: Responder<PromptResponse>,
    cancelled: bool,
    error: Option<String>,
}

/// A request waiting for its session to be installed in the engine.
struct PendingResume {
    session_id: SessionId,
    action: ResumeAction,
}

enum ResumeAction {
    /// A prompt that resumes the session it targets.
    Prompt {
        content: UserMessageContent,
        responder: Responder<PromptResponse>,
    },
    /// A `session/load` that replays the session before it is answered.
    Load {
        responder: Responder<LoadSessionResponse>,
    },
}

impl PendingResume {
    fn session_id(&self) -> &str {
        session_id_text(&self.session_id)
    }

    /// Answers a resume that never completed.
    fn fail(self, error: Error) {
        match self.action {
            ResumeAction::Prompt { responder, .. } => {
                let _ = responder.respond_with_error(error);
            }
            ResumeAction::Load { responder } => {
                let _ = responder.respond_with_error(error);
            }
        }
    }
}

/// The engine's installation of a resumed session.
struct ResumedSession {
    session_id: String,
    messages: Vec<ConversationMessage>,
    model_id: Option<String>,
}

/// A command a client sent, and the client waiting for the engine's report.
struct PendingCommand {
    outcome: CommandOutcome,
    responder: CommandResponder,
}

/// The engine report a client's command is answered from.
///
/// A command is answered by the report that reports its outcome, so each
/// variant stands for the commands that report answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandOutcome {
    /// A session setting, announced through the event reporting its new value.
    Mode,
    Model,
    ReasoningEffort,
    FastMode,
    /// Manual compaction, which reports whether it committed.
    Compaction,
    /// A command that installs a session: `/new` and `/resume`.
    Session,
    /// History navigation: `/undo` and `/redo`.
    History,
}

enum CommandResponder {
    Mode(Responder<SetSessionModeResponse>),
    ConfigOption(Responder<SetSessionConfigOptionResponse>),
    /// A command a client asked for through the slash command it sent as a prompt.
    Prompt(Responder<PromptResponse>),
}

impl CommandResponder {
    /// Answers the client with the state the engine reported.
    fn succeed(self, state: &SessionState) {
        match self {
            Self::Mode(responder) => {
                let _ = responder.respond(SetSessionModeResponse::new());
            }
            Self::ConfigOption(responder) => {
                let _ =
                    responder.respond(SetSessionConfigOptionResponse::new(state.config_options()));
            }
            Self::Prompt(responder) => {
                let _ = responder.respond(PromptResponse::new(StopReason::EndTurn));
            }
        }
    }

    fn fail(self, error: Error) {
        match self {
            Self::Mode(responder) => {
                let _ = responder.respond_with_error(error);
            }
            Self::ConfigOption(responder) => {
                let _ = responder.respond_with_error(error);
            }
            Self::Prompt(responder) => {
                let _ = responder.respond_with_error(error);
            }
        }
    }
}

impl Driver {
    fn new(active_session: String, state: SessionState, locations: SessionLocations) -> Self {
        Self {
            elicitation_forms: false,
            active_session,
            session_issued: false,
            projection: AcpUpdateProjection::new(),
            state,
            turn: None,
            pending_new_session: None,
            pending_resume: None,
            pending_command: None,
            pending_mode_report: false,
            sessions_dir: locations.sessions_dir,
            workspace_dir: locations.workspace_dir,
        }
    }

    async fn drive(
        &mut self,
        connection: &ConnectionTo<Client>,
        ingress: &SessionEngineIngress,
        events: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
        requests: &mut mpsc::UnboundedReceiver<DriverRequest>,
    ) -> Result<(), Error> {
        loop {
            tokio::select! {
                request = requests.recv() => {
                    match request {
                        Some(request) => self.handle_request(connection, ingress, request)?,
                        // Every callback sender is gone, so the connection is over.
                        None => return Ok(()),
                    }
                }
                event = events.recv() => {
                    match event {
                        Some(event) => self.handle_event(connection, ingress, event).await?,
                        None => return Ok(()),
                    }
                }
                () = connection.incoming_closed() => return Ok(()),
            }
        }
    }

    fn handle_request(
        &mut self,
        connection: &ConnectionTo<Client>,
        ingress: &SessionEngineIngress,
        request: DriverRequest,
    ) -> Result<(), Error> {
        match request {
            DriverRequest::NewSession { responder } => {
                self.start_session(connection, ingress, responder)?;
            }
            DriverRequest::Load {
                session_id,
                responder,
            } => {
                self.start_load(ingress, session_id, responder);
            }
            DriverRequest::ListSessions {
                cwd,
                cursor,
                responder,
            } => {
                self.list_sessions(cwd, cursor, responder);
            }
            DriverRequest::Prompt {
                session_id,
                content,
                responder,
            } => {
                self.start_prompt(ingress, session_id, content, responder);
            }
            DriverRequest::SetMode {
                session_id,
                mode_id,
                responder,
            } => {
                self.set_mode(ingress, session_id, mode_id, responder);
            }
            DriverRequest::SetConfigOption {
                session_id,
                config_id,
                value,
                responder,
            } => {
                self.set_config_option(ingress, session_id, config_id, value, responder);
            }
            DriverRequest::Cancel { session_id } => {
                self.cancel(ingress, session_id);
            }
            DriverRequest::Initialized { capabilities } => {
                self.adopt_client_capabilities(&capabilities);
            }
        }
        Ok(())
    }

    fn start_session(
        &mut self,
        connection: &ConnectionTo<Client>,
        ingress: &SessionEngineIngress,
        responder: Responder<NewSessionResponse>,
    ) -> Result<(), Error> {
        if self.turn.is_some() || self.pending_resume.is_some() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InvalidRequest.into(),
                "a prompt turn is already running",
            ));
            return Ok(());
        }
        if self.pending_new_session.is_some() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InvalidRequest.into(),
                "a session is already being created",
            ));
            return Ok(());
        }
        if !self.session_issued {
            self.session_issued = true;
            let _ = responder.respond(self.new_session_response());
            return self.announce_available_commands(connection);
        }
        // The engine owns the session lifecycle, so a new ACP session installs
        // a new engine session and adopts the id the engine reports back.
        if let Err(error) = ingress.submit(SessionCommand::NewSession) {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InternalError.into(),
                error.to_string(),
            ));
            return Ok(());
        }
        self.pending_new_session = Some(responder);
        Ok(())
    }

    /// Starts a `session/load`: the engine installs the session, so the client
    /// is answered once the engine reports the state it restored.
    fn start_load(
        &mut self,
        ingress: &SessionEngineIngress,
        session_id: SessionId,
        responder: Responder<LoadSessionResponse>,
    ) {
        if self.turn.is_some() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InvalidRequest.into(),
                "a prompt turn is already running",
            ));
            return;
        }
        if self.pending_resume.is_some() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InvalidRequest.into(),
                "another session is already being resumed",
            ));
            return;
        }
        let target = session_id_text(&session_id).to_string();
        if let Err(error) = ingress.submit(SessionCommand::ResumeSession(target)) {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InternalError.into(),
                error.to_string(),
            ));
            return;
        }
        self.pending_resume = Some(PendingResume {
            session_id,
            action: ResumeAction::Load { responder },
        });
    }

    /// Lists the sessions a client can load.
    ///
    /// A cold listing reads every transcript it does not have a cached summary
    /// for, so it runs off the connection future that also carries engine
    /// events.
    fn list_sessions(
        &self,
        cwd: Option<PathBuf>,
        cursor: Option<String>,
        responder: Responder<ListSessionsResponse>,
    ) {
        if cursor.is_some() {
            let _ = responder.respond_with_error(invalid_params(
                "letcode lists sessions in one page and issues no cursor",
            ));
            return;
        }
        let sessions_dir = self.sessions_dir.clone();
        let workspace_dir = self.workspace_dir.clone();
        tokio::spawn(async move {
            let listed = tokio::task::spawn_blocking(move || {
                session_infos(&sessions_dir, &workspace_dir, cwd.as_deref())
            })
            .await;
            match listed {
                Ok(Ok(sessions)) => {
                    let _ = responder.respond(ListSessionsResponse::new(sessions));
                }
                Ok(Err(error)) => {
                    let _ = responder.respond_with_error(Error::new(
                        ErrorCode::InternalError.into(),
                        error.to_string(),
                    ));
                }
                Err(error) => {
                    let _ = responder.respond_with_error(Error::new(
                        ErrorCode::InternalError.into(),
                        format!("session listing failed: {error}"),
                    ));
                }
            }
        });
    }

    fn start_prompt(
        &mut self,
        ingress: &SessionEngineIngress,
        session_id: SessionId,
        content: UserMessageContent,
        responder: Responder<PromptResponse>,
    ) {
        if self.turn.is_some() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InvalidRequest.into(),
                "a prompt turn is already running",
            ));
            return;
        }
        if session_id_text(&session_id) != self.active_session {
            // Only one engine session is installed at a time; switching back
            // resumes the transcript that owns this ACP session id.
            if self.pending_resume.is_some() {
                let _ = responder.respond_with_error(Error::new(
                    ErrorCode::InvalidRequest.into(),
                    "another session is already being resumed",
                ));
                return;
            }
            let target = session_id_text(&session_id).to_string();
            if let Err(error) = ingress.submit(SessionCommand::ResumeSession(target.clone())) {
                let _ = responder.respond_with_error(Error::new(
                    ErrorCode::InternalError.into(),
                    error.to_string(),
                ));
                return;
            }
            self.pending_resume = Some(PendingResume {
                session_id,
                action: ResumeAction::Prompt { content, responder },
            });
            return;
        }
        self.submit_prompt(ingress, content, responder);
    }

    /// Starts the turn a prompt asks for, or dispatches the command it carries.
    ///
    /// A command is prompt text and nothing else: content with attachments is
    /// what the client composed, so it keeps reaching the engine as a turn.
    fn submit_prompt(
        &mut self,
        ingress: &SessionEngineIngress,
        content: UserMessageContent,
        responder: Responder<PromptResponse>,
    ) {
        if content.attachments.is_empty() {
            match slash::request(&content.text) {
                SlashRequest::Prompt => {}
                SlashRequest::Command(command) => {
                    self.submit_slash_command(ingress, command, responder);
                    return;
                }
                SlashRequest::Rejected(message) => {
                    let _ = responder.respond_with_error(invalid_params(message));
                    return;
                }
            }
        }
        let submission = UserMessageSubmission::new(submission_id(), content);
        if let Err(error) = ingress.submit(SessionCommand::SubmitPrompt(submission)) {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InternalError.into(),
                error.to_string(),
            ));
            return;
        }
        self.turn = Some(ActiveTurn {
            responder,
            cancelled: false,
            error: None,
        });
    }

    /// Dispatches a slash command and answers it from the engine's report.
    fn submit_slash_command(
        &mut self,
        ingress: &SessionEngineIngress,
        command: SessionCommand,
        responder: Responder<PromptResponse>,
    ) {
        let Some(outcome) = command_outcome(&command) else {
            // Only commands the engine reports are advertised; reaching this
            // means the advertised set and the dispatch table disagree.
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InternalError.into(),
                "letcode cannot report the outcome of that command",
            ));
            return;
        };
        self.submit_command(
            ingress,
            command,
            outcome,
            CommandResponder::Prompt(responder),
        );
    }

    fn set_mode(
        &mut self,
        ingress: &SessionEngineIngress,
        session_id: SessionId,
        mode_id: SessionModeId,
        responder: Responder<SetSessionModeResponse>,
    ) {
        let responder = CommandResponder::Mode(responder);
        if session_id_text(&session_id) != self.active_session {
            responder.fail(unknown_session(&session_id));
            return;
        }
        let Some(mode) = PermissionMode::parse(&mode_id.0) else {
            responder.fail(invalid_params(format!(
                "unknown session mode: {}",
                mode_id.0
            )));
            return;
        };
        self.submit_command(
            ingress,
            SessionCommand::SetPermissionMode(mode),
            CommandOutcome::Mode,
            responder,
        );
    }

    fn set_config_option(
        &mut self,
        ingress: &SessionEngineIngress,
        session_id: SessionId,
        config_id: SessionConfigId,
        value: SessionConfigOptionValue,
        responder: Responder<SetSessionConfigOptionResponse>,
    ) {
        let responder = CommandResponder::ConfigOption(responder);
        if session_id_text(&session_id) != self.active_session {
            responder.fail(unknown_session(&session_id));
            return;
        }
        let Some(value) = value.as_value_id() else {
            responder.fail(invalid_params(format!(
                "session config option '{}' expects a value id",
                config_id.0
            )));
            return;
        };
        let value = value.0.to_string();
        let (command, outcome) = match config_id.0.as_ref() {
            CONFIG_SESSION_MODE => match PermissionMode::parse(&value) {
                Some(mode) => (
                    SessionCommand::SetPermissionMode(mode),
                    CommandOutcome::Mode,
                ),
                None => {
                    responder.fail(invalid_params(format!("unknown session mode: {value}")));
                    return;
                }
            },
            CONFIG_MODEL => (SessionCommand::SetModel(value), CommandOutcome::Model),
            CONFIG_REASONING_EFFORT => (
                SessionCommand::SetReasoningEffort(parse_reasoning_effort(&value)),
                CommandOutcome::ReasoningEffort,
            ),
            other => {
                responder.fail(invalid_params(format!(
                    "unknown session config option: {other}"
                )));
                return;
            }
        };
        self.submit_command(ingress, command, outcome, responder);
    }

    /// Submits a command the engine owns and remembers the client waiting for
    /// it.
    ///
    /// The engine applies these commands asynchronously, so the client is
    /// answered when the engine reports the outcome — or the failure — rather
    /// than from this frontend's own bookkeeping.
    fn submit_command(
        &mut self,
        ingress: &SessionEngineIngress,
        command: SessionCommand,
        outcome: CommandOutcome,
        responder: CommandResponder,
    ) {
        if self.pending_command.is_some() {
            responder.fail(Error::new(
                ErrorCode::InvalidRequest.into(),
                "a session command is already in flight",
            ));
            return;
        }
        if let Err(error) = ingress.submit(command) {
            responder.fail(Error::new(
                ErrorCode::InternalError.into(),
                error.to_string(),
            ));
            return;
        }
        self.pending_command = Some(PendingCommand { outcome, responder });
    }

    /// Answers a pending command once the engine reports an outcome for it.
    ///
    /// `reports` names the commands the report answers: one engine report can
    /// be the outcome of more than one command.
    fn settle_command(&mut self, reports: &[CommandOutcome], failure: Option<Error>) {
        let Some(pending) = self.pending_command.take() else {
            return;
        };
        if !reports.contains(&pending.outcome) {
            self.pending_command = Some(pending);
            return;
        }
        match failure {
            Some(error) => pending.responder.fail(error),
            None => pending.responder.succeed(&self.state),
        }
    }

    fn cancel(&mut self, ingress: &SessionEngineIngress, session_id: SessionId) {
        if session_id_text(&session_id) != self.active_session {
            tracing::debug!(
                session_id = session_id_text(&session_id),
                "ignoring cancel for a session that is not installed"
            );
            return;
        }
        if let Some(turn) = self.turn.as_mut() {
            turn.cancelled = true;
        }
        let _ = ingress.request_interrupt();
    }

    async fn handle_event(
        &mut self,
        connection: &ConnectionTo<Client>,
        ingress: &SessionEngineIngress,
        event: SessionTransportEvent,
    ) -> Result<(), Error> {
        match event {
            SessionTransportEvent::SessionStarted { session_id, .. } => {
                self.adopt_session(connection, session_id)?;
                // `/new` asked for the session the engine just installed.
                self.settle_command(&[CommandOutcome::Session], None);
            }
            SessionTransportEvent::SessionResumed {
                session_id,
                messages,
                model_id,
                ..
            } => {
                self.resume_session(
                    connection,
                    ingress,
                    ResumedSession {
                        session_id,
                        messages,
                        model_id,
                    },
                )?;
                // `/resume` asked for this session, and `/undo` and `/redo`
                // reach the same report because history navigation installs a
                // session rather than continuing one.
                self.settle_command(&[CommandOutcome::Session, CommandOutcome::History], None);
            }
            SessionTransportEvent::PermissionModeChanged { mode } => {
                if self.state.adopt_mode(&mode) {
                    // A session install announces the mode it restored before it
                    // reports the session that runs with it, so the report waits
                    // for that session instead of naming the installed one.
                    if self.session_install_in_flight() {
                        self.pending_mode_report = true;
                    } else {
                        self.report_mode_change(connection, &self.active_session)?;
                    }
                }
                self.settle_command(&[CommandOutcome::Mode], None);
            }
            SessionTransportEvent::ModelChanged { model_id } => {
                self.state.adopt_model(&model_id);
                notify(
                    connection,
                    &self.active_session,
                    self.state.config_option_update(),
                )?;
                self.settle_command(&[CommandOutcome::Model], None);
            }
            SessionTransportEvent::ReasoningEffortChanged { effort } => {
                self.state.adopt_reasoning_effort(effort);
                notify(
                    connection,
                    &self.active_session,
                    self.state.config_option_update(),
                )?;
                self.settle_command(&[CommandOutcome::ReasoningEffort], None);
            }
            // A fast-mode toggle is answered from the value the engine applied.
            SessionTransportEvent::FastModeChanged { .. } => {
                self.settle_command(&[CommandOutcome::FastMode], None);
            }
            SessionTransportEvent::CompactionCommitted { .. } => {
                self.settle_command(&[CommandOutcome::Compaction], None);
            }
            // A compaction that commits nothing states why; the client asked
            // for one, so the reasons are its answer.
            SessionTransportEvent::CompactionNoProgress { blockers } => {
                self.settle_command(
                    &[CommandOutcome::Compaction],
                    Some(compaction_error(&blockers)),
                );
            }
            SessionTransportEvent::CompactionFailed => {
                // The engine reports the failure without naming a reason.
                self.settle_command(&[CommandOutcome::Compaction], Some(compaction_error(&[])));
            }
            // The engine refuses some commands with an informational notice
            // instead of an error event: `/undo` and `/redo` with nowhere to
            // move, `/fast` without a fast mode to toggle, and `/compact` while
            // another turn holds the session. That notice is the report the
            // command receives, so the client is answered with it rather than
            // waiting for a report that never comes. Notices the engine emits
            // about the session itself can arrive while one of these commands is
            // pending; they answer it with their own message.
            SessionTransportEvent::Notice(event) => {
                self.settle_command(
                    &[
                        CommandOutcome::FastMode,
                        CommandOutcome::Compaction,
                        CommandOutcome::History,
                    ],
                    Some(Error::new(ErrorCode::InvalidRequest.into(), event.message)),
                );
            }
            SessionTransportEvent::ModelCatalogUpdated(catalog) => {
                self.state.adopt_catalog(&catalog);
                notify(
                    connection,
                    &self.active_session,
                    self.state.config_option_update(),
                )?;
            }
            SessionTransportEvent::SettingChangeFailed { command } => {
                if let Some(outcome) = command_outcome(&command) {
                    self.settle_command(
                        &[outcome],
                        Some(Error::new(
                            ErrorCode::InvalidRequest.into(),
                            format!("letcode rejected the session {} change", outcome.label()),
                        )),
                    );
                }
            }
            SessionTransportEvent::PermissionRequested { event, handle } => {
                self.ask_permission(connection, event, handle, None).await;
            }
            SessionTransportEvent::ChildPermissionRequested {
                event,
                handle,
                agent_name,
                ..
            } => {
                self.ask_permission(connection, event, handle, agent_name)
                    .await;
            }
            SessionTransportEvent::QuestionRequested { request, handle } => {
                self.ask_question(connection, request, handle).await;
            }
            // A subagent's question is asked of the same client: clients hold
            // the ACP sessions this frontend installs, and a child session is
            // not one of them.
            SessionTransportEvent::ChildQuestionRequested {
                request, handle, ..
            } => {
                self.ask_question(connection, request, handle).await;
            }
            SessionTransportEvent::Error(event) => {
                self.fail_turn(event);
            }
            SessionTransportEvent::Interrupted => {
                if let Some(turn) = self.turn.as_mut() {
                    turn.cancelled = true;
                }
            }
            SessionTransportEvent::Done => {
                self.finish_turn();
                // A manual compaction ends the run it performs with this report,
                // whether or not the engine reported anything else about it.
                self.settle_command(&[CommandOutcome::Compaction], None);
            }
            event => {
                if let Some(update) = self.projection.project(&event) {
                    connection.send_notification(SessionNotification::new(
                        self.active_session.clone(),
                        update,
                    ))?;
                }
            }
        }
        Ok(())
    }

    /// Reports the engine's session, mode, and configuration state.
    ///
    /// The engine owns the session, so the client adopts its id; this frontend
    /// reports the mode and configuration the engine is running with.
    fn new_session_response(&self) -> NewSessionResponse {
        NewSessionResponse::new(self.active_session.clone())
            .modes(self.state.modes())
            .config_options(self.state.config_options())
    }

    fn adopt_session(
        &mut self,
        connection: &ConnectionTo<Client>,
        session_id: String,
    ) -> Result<(), Error> {
        let previous = std::mem::replace(&mut self.active_session, session_id.clone());
        match self.pending_new_session.take() {
            Some(responder) => {
                let _ = responder.respond(self.new_session_response());
            }
            None => {
                tracing::debug!(previous, session_id, "engine switched sessions");
            }
        }
        self.announce_available_commands(connection)
    }

    /// Tells the client which slash commands the installed session dispatches.
    fn announce_available_commands(&self, connection: &ConnectionTo<Client>) -> Result<(), Error> {
        notify(connection, &self.active_session, slash::update())
    }

    /// Whether this frontend is waiting for the engine to install a session.
    ///
    /// `session/load` and a prompt that resumes another session install one, and
    /// the reports answering `/resume`, `/new`, `/undo`, and `/redo` are reports
    /// of installed sessions.
    fn session_install_in_flight(&self) -> bool {
        self.pending_resume.is_some()
            || self
                .pending_command
                .as_ref()
                .is_some_and(|pending| pending.outcome.installs_session())
    }

    /// Adopts the session the engine installed and reports the settings it
    /// restored for it.
    ///
    /// The engine announces a restored mode before it reports the session it
    /// restored, and it reports the route the session runs with in that report,
    /// so both are reported to the session adopted here.
    fn adopt_resumed_session(
        &mut self,
        connection: &ConnectionTo<Client>,
        session_id: &str,
        model_id: Option<&str>,
    ) -> Result<(), Error> {
        self.active_session = session_id.to_string();
        // The client now holds this session, so the session this frontend
        // launched with is no longer the one a later `session/new` serves.
        self.session_issued = true;
        if std::mem::take(&mut self.pending_mode_report) {
            self.report_mode_change(connection, session_id)?;
        }
        if let Some(model_id) = model_id
            && self.state.adopt_model(model_id)
        {
            notify(connection, session_id, self.state.config_option_update())?;
        }
        Ok(())
    }

    /// Adopts the capabilities the client advertised in its handshake.
    fn adopt_client_capabilities(&mut self, capabilities: &ClientCapabilities) {
        self.elicitation_forms = elicitation_form_supported(capabilities);
    }

    /// Adopts the session the engine installed and finishes the request that
    /// asked for it.
    fn resume_session(
        &mut self,
        connection: &ConnectionTo<Client>,
        ingress: &SessionEngineIngress,
        resumed: ResumedSession,
    ) -> Result<(), Error> {
        let ResumedSession {
            session_id,
            messages,
            model_id,
        } = resumed;
        let pending = self.pending_resume.take();
        self.adopt_resumed_session(connection, &session_id, model_id.as_deref())?;
        let Some(pending) = pending else {
            // A command switched the session, so the client learns which one the
            // engine installed and addresses that session from now on.
            return self.announce_available_commands(connection);
        };
        if pending.session_id() != session_id {
            let requested = pending.session_id().to_string();
            pending.fail(Error::new(
                ErrorCode::InvalidParams.into(),
                format!("session {requested} is not available"),
            ));
            return Ok(());
        }
        match pending.action {
            ResumeAction::Prompt { content, responder } => {
                self.submit_prompt(ingress, content, responder);
                Ok(())
            }
            ResumeAction::Load { responder } => self.finish_load(connection, &messages, responder),
        }
    }

    /// Replays the resumed conversation and answers `session/load`.
    ///
    /// The mode and the route the session restored are reported before the
    /// replay, so a client receiving both knows how the session runs while it
    /// replays what happened in it.
    fn finish_load(
        &mut self,
        connection: &ConnectionTo<Client>,
        messages: &[ConversationMessage],
        responder: Responder<LoadSessionResponse>,
    ) -> Result<(), Error> {
        replay_history(connection, &self.active_session, messages)?;
        let _ = responder.respond(self.load_response());
        self.announce_available_commands(connection)
    }

    /// Reports the mode the engine runs with to the session that runs it.
    fn report_mode_change(
        &self,
        connection: &ConnectionTo<Client>,
        session_id: &str,
    ) -> Result<(), Error> {
        notify(connection, session_id, self.state.mode_update())?;
        notify(connection, session_id, self.state.config_option_update())
    }

    /// The state a `session/load` reports for the session it installed.
    fn load_response(&self) -> LoadSessionResponse {
        LoadSessionResponse::new()
            .modes(self.state.modes())
            .config_options(self.state.config_options())
    }

    fn fail_turn(&mut self, event: ErrorEvent) {
        if let Some(pending) = self.pending_resume.take() {
            tracing::debug!(
                session_id = pending.session_id(),
                message = %event.message,
                "session resume failed"
            );
            // The engine installed no session, so the mode it announced while
            // preparing one belongs to no session this client knows.
            self.pending_mode_report = false;
            pending.fail(Error::new(ErrorCode::InternalError.into(), event.message));
            return;
        }
        if let Some(responder) = self.pending_new_session.take() {
            let _ = responder.respond_with_error(Error::new(
                ErrorCode::InternalError.into(),
                event.message.clone(),
            ));
            return;
        }
        match self.turn.as_mut() {
            Some(turn) => {
                if turn.error.is_none() {
                    turn.error = Some(event.message);
                }
            }
            // A command the engine failed is answered here: the engine's reason
            // is its outcome, and no further report is coming.
            None => match self.pending_command.take() {
                Some(pending) => {
                    if pending.outcome.installs_session() {
                        self.pending_mode_report = false;
                    }
                    pending
                        .responder
                        .fail(Error::new(ErrorCode::InternalError.into(), event.message))
                }
                None => {
                    tracing::warn!(message = %event.message, "session engine reported an error")
                }
            },
        }
    }

    fn finish_turn(&mut self) {
        if let Some(turn) = self.turn.take() {
            let ActiveTurn {
                responder,
                cancelled,
                error,
            } = turn;
            match error {
                Some(message) => {
                    let _ = responder
                        .respond_with_error(Error::new(ErrorCode::InternalError.into(), message));
                }
                None => {
                    let stop_reason = if cancelled {
                        StopReason::Cancelled
                    } else {
                        StopReason::EndTurn
                    };
                    let _ = responder.respond(PromptResponse::new(stop_reason));
                }
            }
        }
    }

    /// Asks the client to decide on a tool call and answers the engine handle
    /// with the decision. Every path answers the handle exactly once.
    async fn ask_permission(
        &self,
        connection: &ConnectionTo<Client>,
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
        origin: Option<String>,
    ) {
        let request = RequestPermissionRequest::new(
            self.active_session.clone(),
            permission_tool_call(&event, origin.as_deref()),
            permission_options(&event),
        );
        let decision = match connection.send_request(request).block_task().await {
            Ok(response) => match response.outcome {
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome {
                    option_id, ..
                }) => permission_decision(&option_id.0, event.can_allow_always),
                // The client cancels pending permission requests when the prompt
                // turn is cancelled.
                RequestPermissionOutcome::Cancelled => Decision::Deny,
                other => {
                    tracing::debug!(?other, "unrecognized permission outcome");
                    Decision::Deny
                }
            },
            Err(error) => {
                tracing::warn!(
                    %error,
                    call_id = %event.call_id,
                    "permission request failed; denying the tool call"
                );
                Decision::Deny
            }
        };
        let answered = match decision {
            Decision::AllowOnce => handle.approve(),
            Decision::AllowAlways => handle.allow_always(),
            Decision::Deny => handle.deny(),
        };
        if let Err(error) = answered {
            tracing::warn!(%error, call_id = %event.call_id, "permission response discarded");
        }
    }

    /// Asks the client the questions the question tool raised and answers the
    /// engine handle with what it returned. Every path answers the handle
    /// exactly once.
    ///
    /// A client that has no form to render the questions, that declines them,
    /// or that cancels them without answering leaves them unanswered; this
    /// frontend reports that instead of filling the answers in itself.
    async fn ask_question(
        &self,
        connection: &ConnectionTo<Client>,
        request: QuestionRequest,
        handle: RunnerQuestionRequest,
    ) {
        if !self.elicitation_forms {
            answer_unrouted_question(&handle, QUESTIONS_UNSUPPORTED);
            return;
        }
        let elicitation = question_elicitation(&self.active_session, &request);
        let answered = match connection.send_request(elicitation).block_task().await {
            Ok(response) => match question_answers(&request, response) {
                Ok(answers) => handle.answer(answers),
                Err(reason) => handle.cancel(reason),
            },
            Err(error) => {
                tracing::warn!(%error, "question form failed");
                handle.cancel("letcode could not ask the questions over ACP")
            }
        };
        if let Err(error) = answered {
            tracing::warn!(%error, "question response discarded");
        }
    }
}

impl CommandOutcome {
    /// Whether the report answering this command installs a session.
    fn installs_session(self) -> bool {
        matches!(self, Self::Session | Self::History)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Mode => "mode",
            Self::Model => "model",
            Self::ReasoningEffort => "reasoning effort",
            Self::FastMode => "fast mode",
            Self::Compaction => "context compaction",
            Self::Session => "session",
            Self::History => "session history",
        }
    }
}

/// The engine report that answers `command`.
fn command_outcome(command: &SessionCommand) -> Option<CommandOutcome> {
    match command {
        SessionCommand::SetPermissionMode(_) => Some(CommandOutcome::Mode),
        SessionCommand::SetModel(_) => Some(CommandOutcome::Model),
        SessionCommand::SetReasoningEffort(_) => Some(CommandOutcome::ReasoningEffort),
        SessionCommand::ToggleFastMode => Some(CommandOutcome::FastMode),
        SessionCommand::Compact => Some(CommandOutcome::Compaction),
        SessionCommand::NewSession | SessionCommand::ResumeSession(_) => {
            Some(CommandOutcome::Session)
        }
        SessionCommand::Undo | SessionCommand::Redo => Some(CommandOutcome::History),
        _ => None,
    }
}

/// The failure a manual compaction reports when it commits nothing.
fn compaction_error(blockers: &[String]) -> Error {
    let reason = if blockers.is_empty() {
        String::new()
    } else {
        format!(": {}", blockers.join("; "))
    };
    Error::new(
        ErrorCode::InternalError.into(),
        format!("letcode could not compact the session context{reason}"),
    )
}

fn notify(
    connection: &ConnectionTo<Client>,
    session_id: &str,
    update: SessionUpdate,
) -> Result<(), Error> {
    connection.send_notification(SessionNotification::new(session_id.to_string(), update))
}

/// Replays a resumed conversation so a client that just loaded the session
/// shows the turns it missed.
///
/// A context summary stands for turns the engine retired rather than being one
/// of them, and ACP has no update kind for it, so it is not replayed.
fn replay_history(
    connection: &ConnectionTo<Client>,
    session_id: &str,
    messages: &[ConversationMessage],
) -> Result<(), Error> {
    for message in messages {
        if message.content.is_empty() {
            continue;
        }
        let update = match message.role {
            ConversationRole::User => {
                SessionUpdate::UserMessageChunk(message_chunk(message.content.clone()))
            }
            ConversationRole::Assistant => {
                SessionUpdate::AgentMessageChunk(message_chunk(message.content.clone()))
            }
            ConversationRole::Summary => continue,
        };
        notify(connection, session_id, update)?;
    }
    Ok(())
}

/// Describes every session a client can load.
///
/// Sessions belong to the workspace this frontend serves, so a request that
/// filters on another working directory matches nothing.
fn session_infos(
    sessions_dir: &Path,
    workspace_dir: &Path,
    cwd: Option<&Path>,
) -> anyhow::Result<Vec<SessionInfo>> {
    if cwd.is_some_and(|cwd| cwd != workspace_dir) {
        return Ok(Vec::new());
    }
    Ok(merged_session_summaries(sessions_dir)?
        .into_iter()
        .map(|summary| {
            let mut info = SessionInfo::new(summary.session_id, workspace_dir.to_path_buf());
            if let Some(title) = summary.title {
                info = info.title(title);
            }
            if let Some(timestamp_ms) = summary.last_timestamp_ms {
                info = info.updated_at(iso8601_utc(timestamp_ms));
            }
            info
        })
        .collect())
}

/// Formats an epoch-millisecond instant as an ISO 8601 UTC timestamp.
fn iso8601_utc(timestamp_ms: u128) -> String {
    let seconds = i64::try_from(timestamp_ms / 1000).unwrap_or(i64::MAX);
    let (year, month, day) = civil_date(seconds.div_euclid(86_400));
    let second_of_day = seconds.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3600,
        (second_of_day % 3600) / 60,
        second_of_day % 60
    )
}

/// Converts days since 1970-01-01 into a civil date, using Howard Hinnant's
/// `civil_from_days` so the frontend needs no time-zone database.
fn civil_date(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    let year = year_of_era + era * 400;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn unknown_session(session_id: &SessionId) -> Error {
    invalid_params(format!(
        "session {} is not installed",
        session_id_text(session_id)
    ))
}

fn invalid_params(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidParams.into(), message)
}

enum Decision {
    AllowOnce,
    AllowAlways,
    Deny,
}

/// Maps the option the client returned onto an engine decision.
///
/// `session/request_permission` options are offered from the engine's own
/// policy, so an answer that names an option this request did not offer is
/// denied rather than turned into the grant it names.
fn permission_decision(option_id: &str, repeat_grant_offered: bool) -> Decision {
    match option_id {
        PERMISSION_ALLOW_ONCE => Decision::AllowOnce,
        PERMISSION_ALLOW_ALWAYS if repeat_grant_offered => Decision::AllowAlways,
        _ => Decision::Deny,
    }
}

fn permission_options(event: &PermissionRequestEvent) -> Vec<PermissionOption> {
    let mut options = vec![PermissionOption::new(
        PermissionOptionId::new(PERMISSION_ALLOW_ONCE),
        "Allow once",
        PermissionOptionKind::AllowOnce,
    )];
    if event.can_allow_always {
        let name = event
            .grant_summary
            .clone()
            .unwrap_or_else(|| "Allow always".to_string());
        options.push(PermissionOption::new(
            PermissionOptionId::new(PERMISSION_ALLOW_ALWAYS),
            name,
            PermissionOptionKind::AllowAlways,
        ));
    }
    options.push(PermissionOption::new(
        PermissionOptionId::new(PERMISSION_REJECT),
        "Deny",
        PermissionOptionKind::RejectOnce,
    ));
    options
}

fn permission_tool_call(event: &PermissionRequestEvent, origin: Option<&str>) -> ToolCallUpdate {
    let title = match origin {
        Some(origin) => format!("{origin}: {}", event.summary),
        None => event.summary.clone(),
    };
    ToolCallUpdate::new(
        event.call_id.clone(),
        ToolCallUpdateFields::new()
            .title(title)
            .kind(ToolKind::Other)
            .raw_input(
                event
                    .arguments
                    .as_deref()
                    .and_then(|arguments| serde_json::from_str(arguments).ok()),
            ),
    )
}

/// Answers engine interaction handles this frontend does not route to the
/// client, so the waiting tool call reports the outcome instead of stalling.
fn answer_unrouted_question(handle: &RunnerQuestionRequest, reason: &str) {
    if let Err(error) = handle.cancel(reason) {
        tracing::warn!(%error, "question request discarded");
    }
}

/// Whether the client advertises the form elicitation a question is asked
/// through. A client that advertises only URL elicitation cannot render one.
fn elicitation_form_supported(capabilities: &ClientCapabilities) -> bool {
    capabilities
        .elicitation
        .as_ref()
        .is_some_and(ElicitationCapabilities::supports_form)
}

/// The form that asks the client for the answers a question request needs.
///
/// Every question becomes a required property, in the order the questions were
/// asked. A question accepting one answer maps to a titled string enum and one
/// accepting several to a titled multi-select; both carry the option
/// descriptions the client shows beside the choices.
fn question_elicitation(session_id: &str, request: &QuestionRequest) -> CreateElicitationRequest {
    let mut schema = ElicitationSchema::new();
    for (index, question) in request.questions.iter().enumerate() {
        schema = schema.property(
            question_property_name(index),
            question_property(question),
            true,
        );
    }
    CreateElicitationRequest::new(
        ElicitationFormMode::new(ElicitationSessionScope::new(session_id.to_string()), schema),
        question_message(request),
    )
}

/// The name a question's answer is returned under.
fn question_property_name(index: usize) -> String {
    format!("question_{}", index + 1)
}

/// The message a form shows above its fields.
fn question_message(request: &QuestionRequest) -> String {
    request
        .questions
        .iter()
        .map(|question| question.question.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The form field that collects one question's answer.
fn question_property(question: &QuestionSpec) -> ElicitationPropertySchema {
    let options = question
        .options
        .iter()
        .map(|option| {
            EnumOption::new(option.label.clone(), option.label.clone())
                .description(option.description.clone())
        })
        .collect::<Vec<_>>();
    if question.multiple {
        MultiSelectPropertySchema::titled(options)
            .min_items(1)
            .title(question.header.clone())
            .description(question.question.clone())
            .into()
    } else {
        StringPropertySchema::new()
            .one_of(options)
            .title(question.header.clone())
            .description(question.question.clone())
            .into()
    }
}

/// The engine's answers to a form the client accepted, or the reason the
/// questions stay unanswered.
///
/// A client that declined, cancelled, or answered with an action this frontend
/// does not recognize is reported as such: nothing is filled in on its behalf.
fn question_answers(
    request: &QuestionRequest,
    response: CreateElicitationResponse,
) -> Result<QuestionResponse, String> {
    let action = response.action;
    let ElicitationAction::Accept(accepted) = &action else {
        return Err(elicitation_refusal(&action));
    };
    let answers = request
        .questions
        .iter()
        .enumerate()
        .map(|(index, _)| {
            accepted
                .content
                .as_ref()
                .and_then(|content| content.get(&question_property_name(index)))
                .map(selected_labels)
                .unwrap_or_default()
        })
        .collect();
    Ok(QuestionResponse { answers })
}

/// The labels the client selected for one form field. A field carries a string
/// for a single-choice question and a string array for a multiple-choice one;
/// a value of another type answers nothing.
fn selected_labels(value: &ElicitationContentValue) -> Vec<String> {
    let labels = match value {
        ElicitationContentValue::String(label) => vec![label.clone()],
        ElicitationContentValue::StringArray(labels) => labels.clone(),
        _ => Vec::new(),
    };
    labels
        .into_iter()
        .filter(|label| !label.trim().is_empty())
        .collect()
}

/// The reason the questions stay unanswered when the client did not accept the
/// form.
fn elicitation_refusal(action: &ElicitationAction) -> String {
    match action {
        ElicitationAction::Decline => "the user declined to answer the questions".to_string(),
        ElicitationAction::Cancel => "the client cancelled the questions".to_string(),
        other => format!("the client did not answer the questions ({other:?})"),
    }
}

fn session_id_text(session_id: &SessionId) -> &str {
    &session_id.0
}

fn submission_id() -> String {
    format!(
        "acp-prompt-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    )
}

/// Flattens the text and image blocks of an ACP prompt into the user content
/// the engine accepts. Blocks without an engine representation are reported to
/// the caller instead of being dropped silently.
pub(super) fn prompt_content(blocks: &[ContentBlock]) -> Result<UserMessageContent, Error> {
    let mut text = String::new();
    let mut attachments = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(content) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&content.text);
            }
            ContentBlock::Image(content) => attachments.push(prompt_image(content)?),
            _ => {
                return Err(invalid_params(
                    "letcode accepts text and image prompt content",
                ));
            }
        }
    }
    let content = UserMessageContent::new(text, attachments);
    if content.is_empty() {
        return Err(invalid_params("prompt must contain text or an image"));
    }
    Ok(content)
}

/// Converts an ACP image block into the engine's image attachment. The label is
/// the media type: the protocol carries no file name for prompt images.
fn prompt_image(content: &ImageContent) -> Result<UserImageAttachment, Error> {
    let media_type = content.mime_type.trim();
    if !media_type.starts_with("image/") {
        return Err(invalid_params(format!(
            "prompt image content must be an image type, not {media_type}"
        )));
    }
    let bytes = STANDARD
        .decode(content.data.trim())
        .map_err(|_| invalid_params("prompt image content is not base64"))?;
    Ok(UserImageAttachment::from_bytes(
        media_type, media_type, &bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Duration;

    use agent_client_protocol::schema::v1::{
        AudioContent, ElicitationAcceptAction, ElicitationFormCapabilities, ElicitationMode,
        ElicitationScope, ElicitationUrlCapabilities, MultiSelectItems, OtherElicitationAction,
        ResourceLink, TextContent,
    };
    use agent_client_protocol::{Agent, Channel, Client};
    use tokio::sync::oneshot;

    use crate::acp::session_state::SessionSettings;
    use crate::request_builder::ModelReasoningEffort;
    use crate::tool::QuestionOption;

    /// `bGV0Y29kZQ==` is "letcode".
    const IMAGE_DATA: &str = "bGV0Y29kZQ==";

    fn text_block(text: &str) -> ContentBlock {
        ContentBlock::Text(TextContent::new(text))
    }

    fn image_block(mime_type: &str) -> ContentBlock {
        ContentBlock::Image(ImageContent::new(IMAGE_DATA, mime_type))
    }

    #[test]
    fn prompt_content_joins_text_blocks() {
        let content =
            prompt_content(&[text_block("first"), text_block("second")]).expect("text prompt");
        assert_eq!(content.text, "first\nsecond");
        assert!(content.attachments.is_empty());
    }

    #[test]
    fn prompt_content_converts_image_blocks_into_engine_attachments() {
        let content = prompt_content(&[text_block("look at this"), image_block("image/png")])
            .expect("image prompt");

        assert_eq!(content.text, "look at this");
        assert_eq!(content.attachments.len(), 1);
        assert_eq!(content.attachments[0].mime, "image/png");
        assert_eq!(
            content.attachments[0].data_url,
            "data:image/png;base64,bGV0Y29kZQ=="
        );
        assert_eq!(content.display_text(), "look at this\n[Image: image/png]");
    }

    #[test]
    fn prompt_content_accepts_an_image_without_text() {
        let content = prompt_content(&[image_block("image/jpeg")]).expect("image-only prompt");
        assert!(!content.is_empty());
        assert_eq!(content.attachments.len(), 1);
    }

    #[test]
    fn prompt_content_rejects_content_the_engine_cannot_represent() {
        for block in [
            ContentBlock::Audio(AudioContent::new(IMAGE_DATA, "audio/wav")),
            ContentBlock::ResourceLink(ResourceLink::new("file", "file:///tmp/file")),
        ] {
            let error = prompt_content(&[block]).expect_err("unsupported content");
            assert_eq!(error.code, ErrorCode::InvalidParams);
        }
    }

    #[test]
    fn prompt_content_rejects_media_types_that_are_not_images() {
        let error = prompt_content(&[image_block("application/pdf")])
            .expect_err("non-image media types are unsupported");
        assert_eq!(error.code, ErrorCode::InvalidParams);
    }

    #[test]
    fn prompt_content_rejects_image_data_that_is_not_base64() {
        let block = ContentBlock::Image(ImageContent::new("%%%", "image/png"));
        let error = prompt_content(&[block]).expect_err("invalid base64 is rejected");
        assert_eq!(error.code, ErrorCode::InvalidParams);
    }

    #[test]
    fn prompt_content_rejects_empty_content() {
        assert!(prompt_content(&[text_block("   ")]).is_err());
        assert!(prompt_content(&[]).is_err());
    }

    #[test]
    fn timestamps_format_as_iso8601_utc() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(1_700_000_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(
            iso8601_utc(1_709_164_800_123),
            "2024-02-29T00:00:00Z",
            "a leap day keeps its date"
        );
    }

    #[test]
    fn commands_map_to_the_report_that_answers_them() {
        assert_eq!(
            command_outcome(&SessionCommand::SetPermissionMode(PermissionMode::Yolo)),
            Some(CommandOutcome::Mode)
        );
        assert_eq!(
            command_outcome(&SessionCommand::SetModel("test/model".into())),
            Some(CommandOutcome::Model)
        );
        assert_eq!(
            command_outcome(&SessionCommand::SetReasoningEffort(
                ModelReasoningEffort::High
            )),
            Some(CommandOutcome::ReasoningEffort)
        );
        for (command, outcome) in [
            (SessionCommand::ToggleFastMode, CommandOutcome::FastMode),
            (SessionCommand::Compact, CommandOutcome::Compaction),
            (SessionCommand::NewSession, CommandOutcome::Session),
            (
                SessionCommand::ResumeSession("session-1".into()),
                CommandOutcome::Session,
            ),
            (SessionCommand::Undo, CommandOutcome::History),
            (SessionCommand::Redo, CommandOutcome::History),
        ] {
            assert_eq!(command_outcome(&command), Some(outcome), "{command:?}");
        }
        assert_eq!(
            command_outcome(&SessionCommand::ShowHistoryTree),
            None,
            "a command no engine report answers is not dispatched"
        );
    }

    #[test]
    fn a_compaction_that_commits_nothing_states_why() {
        assert_eq!(
            compaction_error(&[]).message,
            "letcode could not compact the session context"
        );
        assert_eq!(
            compaction_error(&["context is already within budget".to_string()]).message,
            "letcode could not compact the session context: context is already within budget"
        );
    }

    fn permission_request(can_allow_always: bool) -> PermissionRequestEvent {
        let mut event = PermissionRequestEvent::new("call-1", "shell__exec", "Run cargo test");
        event.can_allow_always = can_allow_always;
        event.grant_summary = can_allow_always.then(|| "Allow cargo test always".to_string());
        event
    }

    #[test]
    fn permission_options_offer_repeat_grants_only_when_available() {
        let repeatable = permission_options(&permission_request(true));
        assert_eq!(
            repeatable
                .iter()
                .map(|option| (option.option_id.0.to_string(), option.kind))
                .collect::<Vec<_>>(),
            vec![
                (
                    PERMISSION_ALLOW_ONCE.to_string(),
                    PermissionOptionKind::AllowOnce
                ),
                (
                    PERMISSION_ALLOW_ALWAYS.to_string(),
                    PermissionOptionKind::AllowAlways
                ),
                (
                    PERMISSION_REJECT.to_string(),
                    PermissionOptionKind::RejectOnce
                ),
            ]
        );
        assert_eq!(
            repeatable[1].name, "Allow cargo test always",
            "the grant summary labels the repeat option"
        );

        let once = permission_options(&permission_request(false));
        assert_eq!(once.len(), 2);
        assert!(
            once.iter()
                .all(|option| option.option_id.0.as_ref() != PERMISSION_ALLOW_ALWAYS)
        );
    }

    #[test]
    fn permission_decisions_map_returned_option_ids() {
        assert!(matches!(
            permission_decision(PERMISSION_ALLOW_ONCE, false),
            Decision::AllowOnce
        ));
        assert!(matches!(
            permission_decision(PERMISSION_ALLOW_ALWAYS, true),
            Decision::AllowAlways
        ));
        assert!(matches!(
            permission_decision(PERMISSION_REJECT, true),
            Decision::Deny
        ));
        assert!(matches!(
            permission_decision("future_option", true),
            Decision::Deny
        ));
    }

    #[test]
    fn permission_decisions_reject_grants_this_request_did_not_offer() {
        assert!(
            matches!(
                permission_decision(PERMISSION_ALLOW_ALWAYS, false),
                Decision::Deny
            ),
            "a repeat grant the engine did not offer is not granted"
        );
    }

    #[test]
    fn permission_tool_call_labels_child_requests_with_their_origin() {
        let event = permission_request(true);
        let parent = permission_tool_call(&event, None);
        assert_eq!(parent.fields.title.as_deref(), Some("Run cargo test"));

        let child = permission_tool_call(&event, Some("fixer"));
        assert_eq!(child.fields.title.as_deref(), Some("fixer: Run cargo test"));
        assert_eq!(child.fields.kind, Some(ToolKind::Other));
    }

    /// A question request with one single-choice and one multiple-choice
    /// question, which are the two shapes the question tool validates.
    fn question_request() -> QuestionRequest {
        QuestionRequest {
            questions: vec![
                QuestionSpec {
                    question: "Which parser path should run?".to_string(),
                    header: "Path".to_string(),
                    options: vec![
                        QuestionOption {
                            label: "Fast".to_string(),
                            description: "Skip validation".to_string(),
                        },
                        QuestionOption {
                            label: "Safe".to_string(),
                            description: "Validate everything".to_string(),
                        },
                    ],
                    multiple: false,
                },
                QuestionSpec {
                    question: "Which checks should run?".to_string(),
                    header: "Checks".to_string(),
                    options: vec![
                        QuestionOption {
                            label: "clippy".to_string(),
                            description: "Lints".to_string(),
                        },
                        QuestionOption {
                            label: "test".to_string(),
                            description: "Tests".to_string(),
                        },
                    ],
                    multiple: true,
                },
            ],
        }
    }

    /// The response of a client that filled the form in.
    fn accepted(content: Vec<(&str, ElicitationContentValue)>) -> CreateElicitationResponse {
        CreateElicitationResponse::new(ElicitationAction::Accept(
            ElicitationAcceptAction::new().content(
                content
                    .into_iter()
                    .map(|(name, value)| (name.to_string(), value))
                    .collect::<BTreeMap<_, _>>(),
            ),
        ))
    }

    #[test]
    fn question_forms_need_the_client_to_advertise_form_elicitation() {
        assert!(
            !elicitation_form_supported(&ClientCapabilities::new()),
            "a client that advertises nothing cannot render a form"
        );
        let url_only = ClientCapabilities::new()
            .elicitation(ElicitationCapabilities::new().url(ElicitationUrlCapabilities::new()));
        assert!(
            !elicitation_form_supported(&url_only),
            "a URL elicitation carries no answer to a question"
        );
        let forms = ClientCapabilities::new()
            .elicitation(ElicitationCapabilities::new().form(ElicitationFormCapabilities::new()));
        assert!(elicitation_form_supported(&forms));
    }

    #[test]
    fn a_question_form_carries_a_required_field_per_question() {
        let request = question_elicitation("session-1", &question_request());

        assert_eq!(
            request.message,
            "Which parser path should run?\nWhich checks should run?"
        );
        let ElicitationMode::Form(form) = &request.mode else {
            panic!("expected a form elicitation");
        };
        assert_eq!(
            form.scope,
            ElicitationScope::Session(ElicitationSessionScope::new("session-1".to_string()))
        );

        let schema = &form.requested_schema;
        assert_eq!(schema.properties.len(), 2);
        assert_eq!(
            schema.required,
            Some(vec!["question_1".to_string(), "question_2".to_string()])
        );

        let Some(ElicitationPropertySchema::String(path)) = schema.properties.get("question_1")
        else {
            panic!("expected a single-choice field");
        };
        assert_eq!(path.title.as_deref(), Some("Path"));
        assert_eq!(
            path.description.as_deref(),
            Some("Which parser path should run?")
        );
        assert_eq!(
            path.one_of
                .as_ref()
                .expect("the field offers choices")
                .iter()
                .map(|option| (
                    option.value.as_str(),
                    option.title.as_str(),
                    option.description.as_deref()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("Fast", "Fast", Some("Skip validation")),
                ("Safe", "Safe", Some("Validate everything")),
            ]
        );

        let Some(ElicitationPropertySchema::Array(checks)) = schema.properties.get("question_2")
        else {
            panic!("expected a multiple-choice field");
        };
        assert_eq!(checks.title.as_deref(), Some("Checks"));
        assert_eq!(checks.min_items, Some(1), "a question needs an answer");
        let MultiSelectItems::Titled(items) = &checks.items else {
            panic!("expected titled choices");
        };
        assert_eq!(
            items
                .options
                .iter()
                .map(|option| (option.value.as_str(), option.description.as_deref()))
                .collect::<Vec<_>>(),
            vec![("clippy", Some("Lints")), ("test", Some("Tests"))]
        );
    }

    #[test]
    fn an_accepted_form_returns_the_selected_labels() {
        let answers = question_answers(
            &question_request(),
            accepted(vec![
                (
                    "question_1",
                    ElicitationContentValue::String("Safe".to_string()),
                ),
                (
                    "question_2",
                    ElicitationContentValue::StringArray(vec![
                        "clippy".to_string(),
                        "test".to_string(),
                    ]),
                ),
            ]),
        )
        .expect("the client filled the form in");
        assert_eq!(
            answers.answers,
            vec![
                vec!["Safe".to_string()],
                vec!["clippy".to_string(), "test".to_string()]
            ]
        );
    }

    #[test]
    fn an_accepted_form_answers_only_with_labels_the_client_chose() {
        let answers = question_answers(
            &question_request(),
            accepted(vec![
                ("question_1", ElicitationContentValue::Boolean(true)),
                (
                    "question_2",
                    ElicitationContentValue::StringArray(vec!["  ".to_string()]),
                ),
                ("question_9", ElicitationContentValue::from("stray")),
            ]),
        )
        .expect("the client filled the form in");
        assert_eq!(answers.answers, vec![Vec::<String>::new(), Vec::new()]);
    }

    #[test]
    fn a_declined_or_cancelled_form_leaves_the_questions_unanswered() {
        let request = question_request();
        for (action, reason) in [
            (
                ElicitationAction::Decline,
                "the user declined to answer the questions",
            ),
            (
                ElicitationAction::Cancel,
                "the client cancelled the questions",
            ),
        ] {
            let reported = question_answers(&request, CreateElicitationResponse::new(action))
                .expect_err("the client did not answer");
            assert_eq!(reported, reason);
        }

        let unknown =
            ElicitationAction::Other(OtherElicitationAction::new("teleport", BTreeMap::new()));
        let reported = question_answers(&request, CreateElicitationResponse::new(unknown))
            .expect_err("an unrecognized action answers nothing");
        assert!(reported.contains("teleport"), "{reported}");
    }

    /// A driver that owns a session and no engine, for the client round trips a
    /// test exercises directly.
    fn driver(elicitation_forms: bool) -> Driver {
        let settings = SessionSettings::new(Vec::new(), None);
        let mut driver = Driver::new(
            "session-1".to_string(),
            SessionState::new("auto", "test/model", &settings),
            SessionLocations {
                sessions_dir: PathBuf::from("sessions"),
                workspace_dir: PathBuf::from("workspace"),
            },
        );
        driver.elicitation_forms = elicitation_forms;
        driver
    }

    /// Asks the fixture questions over a real ACP connection and reports what
    /// the engine handle was answered with together with the form the client
    /// received.
    ///
    /// The client replies to every form with `answer`, which stays unused when
    /// the driver asks it nothing.
    async fn ask_over_acp(
        elicitation_forms: bool,
        answer: CreateElicitationResponse,
    ) -> (
        Result<QuestionResponse, String>,
        Option<CreateElicitationRequest>,
    ) {
        let (agent_transport, client_transport) = Channel::duplex();
        let (answers_tx, answers_rx) = oneshot::channel();
        let (forms_tx, mut forms_rx) = mpsc::unbounded_channel();
        let agent = Agent
            .builder()
            .connect_with(agent_transport, async move |connection| {
                driver(elicitation_forms)
                    .ask_question(
                        &connection,
                        question_request(),
                        RunnerQuestionRequest::new(answers_tx),
                    )
                    .await;
                Ok(())
            });
        let client = Client
            .builder()
            .on_receive_request(
                {
                    let forms_tx = forms_tx.clone();
                    async move |request: CreateElicitationRequest, responder, _connection| {
                        forms_tx
                            .send(request)
                            .map_err(agent_client_protocol::Error::into_internal_error)?;
                        responder.respond(answer.clone())?;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(client_transport, async move |connection| {
                // The client stays connected until the agent has the round trip
                // it asked for, rather than closing under it.
                connection.incoming_closed().await;
                Ok(())
            });
        let (agent_result, client_result) = tokio::join!(agent, client);
        agent_result.expect("ACP agent connection failed");
        client_result.expect("ACP client connection failed");
        let answers = answers_rx.await.expect("the question handle was answered");
        (answers, forms_rx.try_recv().ok())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_question_the_client_answers_is_asked_as_a_form() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let (answered, form) = ask_over_acp(
                true,
                accepted(vec![
                    (
                        "question_1",
                        ElicitationContentValue::String("Safe".to_string()),
                    ),
                    (
                        "question_2",
                        ElicitationContentValue::StringArray(vec!["test".to_string()]),
                    ),
                ]),
            )
            .await;
            assert_eq!(
                answered.expect("the client answered the questions").answers,
                vec![vec!["Safe".to_string()], vec!["test".to_string()]]
            );
            let form = form.expect("the client received a form");
            assert_eq!(
                form.message,
                "Which parser path should run?\nWhich checks should run?"
            );
        })
        .await
        .expect("the question round trip timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_declined_form_leaves_the_questions_unanswered() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let (answered, _) = ask_over_acp(
                true,
                CreateElicitationResponse::new(ElicitationAction::Decline),
            )
            .await;
            assert_eq!(
                answered.expect_err("the client declined"),
                "the user declined to answer the questions"
            );
        })
        .await
        .expect("the question round trip timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_client_without_form_elicitation_is_refused_explicitly() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let (answered, form) = ask_over_acp(
                false,
                CreateElicitationResponse::new(ElicitationAction::Decline),
            )
            .await;
            assert_eq!(
                answered.expect_err("the questions were refused"),
                QUESTIONS_UNSUPPORTED
            );
            assert!(form.is_none(), "the client was asked nothing");
        })
        .await
        .expect("the question round trip timed out");
    }
}
