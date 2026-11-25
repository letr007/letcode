use super::*;

fn install_test_redactor(model_dir: &str) -> (tempfile::TempDir, RedactGlobals) {
    let config = tempfile::tempdir().unwrap();
    let target = config.path().join("models/astrlink-guard");
    fs::create_dir_all(&target).unwrap();
    for name in ["model_int8.onnx", "model_int8.onnx_data", "tokenizer.json"] {
        fs::copy(PathBuf::from(model_dir).join(name), target.join(name)).unwrap();
    }
    let globals = RedactGlobals {
        active: crate::redact::active_snapshot(),
    };
    crate::redact::install(true, config.path(), &config.path().join("sessions"));
    (config, globals)
}

#[test]
#[ignore = "requires LETCODE_PII_MODEL_DIR pointing to the local PII model"]
fn mixed_history_keeps_its_representation_in_requests_and_journal() {
    let model_dir =
        std::env::var("LETCODE_PII_MODEL_DIR").expect("LETCODE_PII_MODEL_DIR is required");
    let _lock = REDACT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (config, _globals) = install_test_redactor(&model_dir);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(180), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let (requests_tx, mut requests_rx) = tokio::sync::mpsc::unbounded_channel();
            let server = tokio::spawn(async move {
                for _ in 0..3 {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let (_, body) = read_test_http_request(&mut stream).await;
                    serve_anthropic_text_completion(&mut stream, "reply@example.com").await;
                    requests_tx.send(body).unwrap();
                }
            });
            let sessions = config.path().join("sessions");
            let transcript = Arc::new(Mutex::new(
                crate::transcript::TranscriptRecorder::create(&sessions).unwrap(),
            ));
            let mut agent = test_agent();
            install_active_epoch_route_at(
                &mut agent,
                ApiProtocol::Anthropic,
                false,
                &format!("http://{address}/v1"),
            );
            agent.prelude = vec![PromptMessage::system("Do not change system@example.com")];
            crate::session::engine::rehydrate_agent_from_transcript(&mut agent, &transcript)
                .unwrap();
            let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
            let runner = crate::session::AgentRunner::with_transcript(event_tx, transcript.clone());
            runner
                .run_prompt(
                    &mut agent,
                    crate::user_content::UserMessageSubmission::new(
                        "masked",
                        UserMessageContent::from("Contact alice@example.com"),
                    ),
                )
                .await
                .unwrap();
            let mut main_requests = 0;
            for _ in 0..2 {
                let request = requests_rx.recv().await.unwrap();
                assert!(!request.contains("alice@example.com"), "{request}");
                assert!(request.contains("{|>EMAIL:"), "{request}");
                main_requests += usize::from(request.contains("system@example.com"));
            }
            assert_eq!(main_requests, 1);
            loop {
                let event = event_rx
                    .recv()
                    .await
                    .expect("title event channel remains open");
                if matches!(
                    event,
                    crate::session::SessionTransportEvent::SessionTitleUpdated { .. }
                ) {
                    break;
                }
            }
            let on_history = agent.active_history_items();
            assert!(
                !serde_json::to_string(&on_history)
                    .unwrap()
                    .contains("reply@example.com")
            );
            let journal = fs::read_to_string(transcript.lock().unwrap().path()).unwrap();
            assert!(!journal.contains("alice@example.com"));
            assert!(!journal.contains("reply@example.com"));

            crate::redact::install(false, config.path(), &sessions);
            runner
                .run_prompt(
                    &mut agent,
                    crate::user_content::UserMessageSubmission::new(
                        "plain",
                        UserMessageContent::from("Contact bob@example.com"),
                    ),
                )
                .await
                .unwrap();
            let request = requests_rx.recv().await.unwrap();
            assert!(request.contains("bob@example.com"));
            assert!(!request.contains("alice@example.com"));
            assert!(request.contains("{|>EMAIL:"));
            server.await.unwrap();
            let before = serde_json::to_value(agent.active_history_items()).unwrap();
            crate::redact::install(true, config.path(), &sessions);
            crate::session::engine::rehydrate_agent_from_transcript(&mut agent, &transcript)
                .unwrap();
            assert_eq!(
                serde_json::to_value(agent.active_history_items()).unwrap(),
                before
            );
            let session = transcript.lock().unwrap().session_id().to_string();
            let first = match &agent.active_history_items()[0] {
                HistoryItem::UserMessage { content } => content.text.clone(),
                _ => panic!("user message"),
            };
            assert_eq!(
                crate::redact::restore_session_text(&session, &first),
                "Contact alice@example.com"
            );
        })
        .await
        .expect("bounded runner test");
    });
}

#[test]
#[ignore = "requires LETCODE_PII_MODEL_DIR pointing to the local PII model"]
fn tool_history_preserves_json_types_and_the_prepared_output() {
    let model_dir =
        std::env::var("LETCODE_PII_MODEL_DIR").expect("LETCODE_PII_MODEL_DIR is required");
    let _lock = REDACT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (config, _globals) = install_test_redactor(&model_dir);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let mut agent = test_agent();
        agent.runtime_snapshot.session_id = Some("tool-history".into());
        install_active_epoch_route(&mut agent, ApiProtocol::Anthropic);
        let arguments = serde_json::json!({
            "text": "alice@example.com", "phone": 13800138000u64, "enabled": true,
            "items": [null, "bob@example.com"], "key@example.com": 42,
        });
        let call = test_tool_call("util__echo", &arguments.to_string());
        agent
            .append_history_item(HistoryItem::user("inspect values"))
            .unwrap();
        agent
            .append_history_item(HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
                calls: vec![call.clone()],
            })
            .unwrap();
        let calls = match &agent.active_history_items()[1] {
            HistoryItem::AssistantTurn { calls, .. } => calls.clone(),
            _ => panic!("assistant turn"),
        };
        let mut stored: Value = serde_json::from_str(&calls[0].arguments_json).unwrap();
        assert_eq!(stored["phone"], arguments["phone"]);
        assert_eq!(stored["enabled"], true);
        assert_eq!(stored["key@example.com"], 42);
        assert!(stored["items"][0].is_null());
        assert_ne!(stored["text"], arguments["text"]);
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments_json).unwrap(),
            arguments
        );
        let restore = crate::redact::session_restore_map(
            &crate::redact::installed().unwrap(),
            agent.session_id(),
        )
        .unwrap();
        crate::redact::restore_value(&restore, &mut stored);
        assert_eq!(stored, arguments);

        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .unwrap();
        assert!(record.output.ok);
        assert_eq!(
            record.output.data.as_ref().unwrap()["result"],
            arguments["text"]
        );
        let prepared = record.history_output.clone().unwrap();
        crate::redact::install(false, config.path(), &config.path().join("sessions"));
        agent
            .record_tool_call_result(&call, record, &mut |_| std::future::ready(Ok(())))
            .await
            .unwrap();
        let output = match agent.active_history_items().last().unwrap() {
            HistoryItem::ToolOutput { output_json, .. } => {
                serde_json::from_str::<Value>(output_json).unwrap()
            }
            _ => panic!("tool output"),
        };
        assert_eq!(
            output,
            serde_json::to_value(prepared.for_text_history()).unwrap()
        );
        assert!(events.iter().any(|event| matches!(event,
            AgentEvent::ToolCallFinished { output, .. } if output.data == prepared.data
        )));
        let route = agent.resolved_model_route().unwrap();
        let build = build_request_with_policy(
            RequestBuilderInput {
                model_id: agent.model(),
                model: agent.active_model_metadata(),
                prelude: &[],
                snapshot: &agent.runtime_snapshot,
                tools: &[],
            },
            None,
        )
        .unwrap();
        let input = crate::model_runtime::projection::model_request_from_prompt_plan(
            route,
            &agent.active_model_metadata(),
            &build.prompt_plan,
            &[],
        )
        .unwrap();
        route.binding.prepare_request(&input).unwrap();
    });
}
