use super::*;
use std::collections::BTreeMap;
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    AudioContent, ElicitationAcceptAction, ElicitationFormCapabilities, ElicitationMode,
    ElicitationScope, ElicitationUrlCapabilities, MultiSelectItems, OtherElicitationAction,
    PromptRequest, ResourceLink, TextContent,
};
use agent_client_protocol::{Agent, Channel, Client};
use tokio::sync::oneshot;

use crate::acp::session_state::SessionSettings;
use crate::request_builder::ModelReasoningEffort;
use crate::session::TurnOutcome;
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

    let Some(ElicitationPropertySchema::String(path)) = schema.properties.get("question_1") else {
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

    let Some(ElicitationPropertySchema::Array(checks)) = schema.properties.get("question_2") else {
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
            let (_engine, ingress, _egress) = SessionEngine::new();
            driver(elicitation_forms)
                .handle_event(
                    &connection,
                    &ingress,
                    SessionTransportEvent::QuestionRequested {
                        request: question_request(),
                        handle: RunnerQuestionRequest::new(answers_tx),
                    },
                )
                .await?;
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

mod integration_tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn manual_compaction_answers_only_after_engine_acceptance() {
        for result in [
            SessionTransportEvent::CompactionCommitted { summary: None },
            SessionTransportEvent::Error(ErrorEvent::new("missing credentials")),
            SessionTransportEvent::Notice(crate::session::NoticeEvent::info("Turn still running")),
        ] {
            let succeeds = matches!(result, SessionTransportEvent::CompactionCommitted { .. });
            tokio::time::timeout(Duration::from_secs(30), async {
                let (agent_transport, client_transport) = Channel::duplex();
                let (requests_tx, mut requests_rx) = mpsc::unbounded_channel();
                let agent = Agent
                    .builder()
                    .on_receive_request(
                        async move |request: PromptRequest, responder, _connection| {
                            requests_tx
                                .send(DriverRequest::Prompt {
                                    session_id: request.session_id,
                                    content: prompt_content(&request.prompt)?,
                                    responder,
                                })
                                .map_err(Error::into_internal_error)?;
                            Ok(())
                        },
                        agent_client_protocol::on_receive_request!(),
                    )
                    .connect_with(agent_transport, async move |connection| {
                        let (_engine, ingress, _egress) = SessionEngine::new();
                        let mut driver = driver(false);
                        let request = requests_rx.recv().await.unwrap();
                        driver.handle_request(&connection, &ingress, request)?;
                        for event in [
                            SessionTransportEvent::CompactionCommitted { summary: None },
                            SessionTransportEvent::CompactionNoProgress {
                                blockers: vec!["automatic".into()],
                            },
                            SessionTransportEvent::CompactionFailed,
                            SessionTransportEvent::Notice(crate::session::NoticeEvent::info(
                                "peer notice",
                            )),
                            SessionTransportEvent::Error(ErrorEvent::new("peer error")),
                            SessionTransportEvent::Done,
                        ] {
                            driver.handle_event(&connection, &ingress, event).await?;
                            assert!(
                                driver
                                    .pending_command
                                    .as_ref()
                                    .is_some_and(|pending| !pending.accepted)
                            );
                        }
                        driver
                            .handle_event(
                                &connection,
                                &ingress,
                                SessionTransportEvent::ManualCompactionStarted,
                            )
                            .await?;
                        assert!(driver.pending_command.as_ref().unwrap().accepted);
                        driver.handle_event(&connection, &ingress, result).await?;
                        driver
                            .handle_event(&connection, &ingress, SessionTransportEvent::Done)
                            .await?;
                        assert!(driver.pending_command.is_none());
                        connection.incoming_closed().await;
                        Ok(())
                    });
                let client =
                    Client
                        .builder()
                        .connect_with(client_transport, async move |connection| {
                            let response = connection
                                .send_request(PromptRequest::new(
                                    SessionId::new("session-1"),
                                    vec![text_block("/compact")],
                                ))
                                .block_task()
                                .await;
                            assert_eq!(response.is_ok(), succeeds);
                            Ok(())
                        });
                let (agent_result, client_result) = tokio::join!(agent, client);
                agent_result.unwrap();
                client_result.unwrap();
            })
            .await
            .expect("manual compaction ACP response timed out");
        }
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

    /// An interrupted run answers the prompt the client is waiting on.
    ///
    /// The engine reports an interrupt with `Interrupted` and never follows it
    /// with `Done`. A frontend that settles turns only on `Done` leaves that
    /// prompt unanswered, and the client's next prompts queue behind it forever.
    #[tokio::test(flavor = "current_thread")]
    async fn an_interrupted_turn_is_answered_as_cancelled() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let (agent_transport, client_transport) = Channel::duplex();
            let (requests_tx, mut requests_rx) = mpsc::unbounded_channel();

            let agent = Agent
                .builder()
                .on_receive_request(
                    {
                        let requests_tx = requests_tx.clone();
                        async move |request: PromptRequest, responder, _connection| {
                            let content = prompt_content(&request.prompt)?;
                            requests_tx
                                .send(DriverRequest::Prompt {
                                    session_id: request.session_id,
                                    content,
                                    responder,
                                })
                                .map_err(Error::into_internal_error)?;
                            Ok(())
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_with(agent_transport, async move |connection| {
                    let (mut engine, ingress, _egress) = SessionEngine::new();
                    let mut driver = driver(false);
                    let request = requests_rx.recv().await.expect("the client sent a prompt");
                    driver.handle_request(&connection, &ingress, request)?;
                    driver
                        .handle_event(
                            &connection,
                            &ingress,
                            SessionTransportEvent::Error(ErrorEvent::new("peer failed")),
                        )
                        .await?;
                    driver
                        .handle_event(&connection, &ingress, SessionTransportEvent::Done)
                        .await?;
                    assert!(
                        driver
                            .turn
                            .as_ref()
                            .is_some_and(|turn| !turn.accepted && turn.error.is_none())
                    );
                    let Some(crate::session::engine::SessionEngineControl::Command(
                        crate::session::engine::SessionEngineCommand::Prompt(prompt),
                    )) = engine.recv_control().await
                    else {
                        panic!("expected prompt command");
                    };
                    driver.cancel(&ingress, SessionId::new("session-1"));
                    assert!(driver.turn.is_some());
                    assert!(matches!(
                        engine.recv_control().await,
                        Some(crate::session::engine::SessionEngineControl::Command(
                            crate::session::engine::SessionEngineCommand::CancelPrompt(id)
                        )) if id == prompt.id
                    ));
                    driver
                        .handle_event(
                            &connection,
                            &ingress,
                            SessionTransportEvent::QueuedPromptAccepted { prompt },
                        )
                        .await?;
                    driver
                        .handle_event(
                            &connection,
                            &ingress,
                            SessionTransportEvent::TurnFinished {
                                outcome: TurnOutcome::Interrupted,
                            },
                        )
                        .await?;
                    driver.cancel(&ingress, SessionId::new("session-1"));
                    assert!(matches!(
                        engine.try_recv_control(),
                        Err(mpsc::error::TryRecvError::Empty)
                    ));
                    connection.incoming_closed().await;
                    Ok(())
                });

            let client = Client
                .builder()
                .connect_with(client_transport, async move |connection| {
                    let response = connection
                        .send_request(PromptRequest::new(
                            SessionId::new("session-1"),
                            vec![text_block("hello")],
                        ))
                        .block_task()
                        .await
                        .expect("the interrupted turn was answered");
                    assert_eq!(response.stop_reason, StopReason::Cancelled);
                    Ok(())
                });

            let (agent_result, client_result) = tokio::join!(agent, client);
            agent_result.expect("ACP agent connection failed");
            client_result.expect("ACP client connection failed");
        })
        .await
        .expect("the interrupted turn timed out");
    }
}
