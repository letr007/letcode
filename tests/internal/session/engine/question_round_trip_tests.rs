use crate::tool::{QuestionOption, QuestionRequest, QuestionResponse, QuestionSpec};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinSet;

const QUESTION_CALL_ID: &str = "call-question";

fn expected_question_request() -> QuestionRequest {
    QuestionRequest {
        questions: vec![QuestionSpec {
            question: "Which runtime should this session use?".into(),
            header: "Runtime".into(),
            options: vec![
                QuestionOption {
                    label: "Local".into(),
                    description: "Use the local runtime".into(),
                },
                QuestionOption {
                    label: "Remote".into(),
                    description: "Use the remote runtime".into(),
                },
            ],
            multiple: false,
        }],
    }
}

async fn read_question_stub_request(socket: &mut tokio::net::TcpStream) -> serde_json::Value {
    let mut request = Vec::new();
    let mut chunk = [0u8; 4096];
    let (header_end, content_length) = loop {
        let read = socket.read(&mut chunk).await.expect("stub request read");
        assert!(read > 0, "stub request closed before headers");
        request.extend_from_slice(&chunk[..read]);
        assert!(request.len() <= 2 * 1024 * 1024, "stub request too large");
        let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
            continue;
        };
        let header_end = at + 4;
        let headers =
            std::str::from_utf8(&request[..header_end]).expect("request headers are utf-8");
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("content length"))
            })
            .expect("responses request has content length");
        assert!(content_length <= 2 * 1024 * 1024, "stub body too large");
        break (header_end, content_length);
    };
    while request.len() < header_end + content_length {
        let read = socket.read(&mut chunk).await.expect("stub body read");
        assert!(read > 0, "stub request closed before body");
        request.extend_from_slice(&chunk[..read]);
    }
    serde_json::from_slice(&request[header_end..header_end + content_length])
        .expect("stub request body is json")
}

async fn write_question_stub_events(
    socket: &mut tokio::net::TcpStream,
    events: &[serde_json::Value],
) {
    let body: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("stub response write");
}

async fn serve_question_stub(listener: tokio::net::TcpListener) -> Vec<serde_json::Value> {
    let arguments = serde_json::to_string(&expected_question_request()).expect("question args");
    let mut bodies = Vec::new();
    for index in 0..2 {
        let (mut socket, _) = listener.accept().await.expect("stub accept");
        bodies.push(read_question_stub_request(&mut socket).await);
        let events = if index == 0 {
            vec![
                serde_json::json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-question","call_id":QUESTION_CALL_ID,"name":"question"}}),
                serde_json::json!({"type":"response.function_call_arguments.done","item_id":"item-question","arguments":arguments}),
                serde_json::json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-question","call_id":QUESTION_CALL_ID,"name":"question","arguments":arguments}}),
                serde_json::json!({"type":"response.completed","response":{"id":"resp-question","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}),
            ]
        } else {
            vec![
                serde_json::json!({"type":"response.output_text.delta","delta":"Answers received"}),
                serde_json::json!({"type":"response.completed","response":{"id":"resp-answer","status":"completed","usage":{"input_tokens":14,"output_tokens":4,"total_tokens":18}}}),
            ]
        };
        write_question_stub_events(&mut socket, &events).await;
    }
    bodies
}

#[tokio::test]
async fn question_tool_round_trip_returns_answers_to_the_model_and_completes_the_turn() {
    tokio::time::timeout(Duration::from_secs(30), async {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind stub listener");
        let address = listener.local_addr().expect("stub address");
        let mut stub_tasks = JoinSet::new();
        stub_tasks.spawn(serve_question_stub(listener));

        let dir = tempfile::tempdir().expect("temp dir");
        let config_path = dir.path().join("letcode.toml");
        let sessions = dir.path().join("sessions");
        fs::write(
            &config_path,
            format!(
                r#"active_provider = "test"
[global]
redact = false
sessions_dir = "{}"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model.capabilities]
tools = true
"#,
                toml_path(&sessions)
            ),
        )
        .expect("write config");
        let config = AppConfig::load_from_path(&config_path).expect("load config");
        let mut recorder = TranscriptRecorder::create(&sessions).expect("create transcript");
        recorder
            .record_session_started("test/model")
            .expect("record session start");
        recorder
            .record_session_title("Question round trip")
            .expect("record session title");
        recorder
            .record_subagents_policy(&config.subagents)
            .expect("record subagents policy");
        let transcript = Arc::new(StdMutex::new(recorder));
        let factory = Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
            config.providers.clone(),
            config.global.retry.clone(),
            config.runtime_catalog.clone(),
        ));
        let mut agent = Agent::new("model", 3, 6);
        agent.apply_prepared_route(
            factory
                .prepare_route(ModelRoute::new("test", "model"))
                .expect("prepare route"),
        );
        agent.set_primary_route_factory(factory);
        agent.set_subagents_policy(config.subagents.clone());
        crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
        let mut settings =
            crate::session_engine_config(&config, Default::default(), String::new(), None);
        settings
            .route_api_key_configured
            .insert("test/model".into(), true);
        let runtime = SubagentPool::new();
        let (_mcp, mcp_rx) = mpsc::unbounded_channel();
        let (_reload, reload_rx) = mpsc::unbounded_channel();
        let (control, control_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let (title, _titles) = mpsc::unbounded_channel();
        let mut engine_tasks = JoinSet::new();
        engine_tasks.spawn(run_engine_loop(
            agent,
            transcript.clone(),
            settings,
            None,
            mcp_rx,
            reload_rx,
            control_rx,
            control.clone(),
            events.clone(),
            title,
            runtime,
            Default::default(),
            events,
        ));
        control
            .send(SessionEngineControl::Command(SessionEngineCommand::Prompt(
                crate::user_content::UserMessageSubmission::new(
                    "ask me a question",
                    crate::user_content::UserMessageContent::from("ask me a question"),
                ),
            )))
            .expect("submit prompt");

        let mut question_requests = 0;
        loop {
            match event_rx.recv().await.expect("engine event channel closed") {
                SessionTransportEvent::QuestionRequested { request, handle } => {
                    assert_eq!(request, expected_question_request());
                    question_requests += 1;
                    handle
                        .answer(QuestionResponse {
                            answers: vec![vec!["Local".into()]],
                        })
                        .expect("question answer accepted");
                }
                SessionTransportEvent::TurnFinished { outcome } => {
                    assert_eq!(outcome, crate::session::TurnOutcome::Completed);
                    break;
                }
                SessionTransportEvent::Error(error) => panic!("{}", error.message),
                _ => {}
            }
        }
        assert_eq!(question_requests, 1);

        control
            .send(SessionEngineControl::Shutdown)
            .expect("shutdown engine");
        engine_tasks
            .join_next()
            .await
            .expect("engine task")
            .expect("engine task completed");
        let bodies = stub_tasks
            .join_next()
            .await
            .expect("stub task")
            .expect("stub task completed");

        assert_eq!(bodies.len(), 2);
        let input = bodies[1]
            .get("input")
            .and_then(serde_json::Value::as_array)
            .expect("second request carries input items");
        let outputs: Vec<_> = input
            .iter()
            .filter(|item| {
                item.get("type").and_then(serde_json::Value::as_str) == Some("function_call_output")
            })
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0]
                .get("call_id")
                .and_then(serde_json::Value::as_str),
            Some(QUESTION_CALL_ID)
        );
        let output: serde_json::Value = serde_json::from_str(
            outputs[0]
                .get("output")
                .and_then(serde_json::Value::as_str)
                .expect("text tool output"),
        )
        .expect("question tool output is json");
        assert_eq!(
            output.get("tool").and_then(serde_json::Value::as_str),
            Some("question")
        );
        assert_eq!(output["ok"], true);
        assert_eq!(output["data"]["answers"], serde_json::json!([["Local"]]));
        assert_eq!(
            output["data"]["message"],
            "User has answered your questions:\n- Runtime: Local"
        );

        let records = read_records(transcript.lock().unwrap().path()).expect("read records");
        let finished = records
            .iter()
            .find_map(|record| match &record.event {
                TranscriptEvent::ToolCallFinished {
                    call_id,
                    name,
                    ok,
                    output,
                } if call_id == QUESTION_CALL_ID => Some((name, ok, output)),
                _ => None,
            })
            .expect("question tool completion is persisted");
        assert_eq!(finished.0, "question");
        assert!(*finished.1);
        let data = finished.2.data.as_ref().expect("question output data");
        assert_eq!(data.get("answers"), Some(&serde_json::json!([["Local"]])));
        assert!(records.iter().any(|record| matches!(
            &record.event,
            TranscriptEvent::TurnFinalized(event) if event.outcome == "completed"
        )));
    })
    .await
    .expect("question round trip completed in time");
}
