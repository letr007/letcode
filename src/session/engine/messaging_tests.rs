#[tokio::test]
async fn idle_root_messages_use_the_engine_user_entry_and_stop_with_the_root() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    tokio::time::timeout(Duration::from_secs(5), async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for index in 0..6 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let body = loop {
                    let mut chunk = [0u8; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&chunk[..read]);
                    if let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        let length = std::str::from_utf8(&request[..at]).unwrap().lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().unwrap())
                        }).unwrap();
                        if request.len() >= at + 4 + length { break serde_json::from_slice::<serde_json::Value>(&request[at + 4..at + 4 + length]).unwrap(); }
                    }
                };
                let input = body["input"].to_string();
                assert_eq!(input.contains("idle member message"), index != 0 && index != 3);
                assert!(!input.contains("late failure"));
                let mut events = if index == 2 {
                    let arguments = serde_json::json!({"objective":"resumed child"}).to_string();
                    vec![
                        serde_json::json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-child","call_id":"call-child","name":"agent__explore"}}),
                        serde_json::json!({"type":"response.function_call_arguments.done","item_id":"item-child","arguments":arguments}),
                        serde_json::json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-child","call_id":"call-child","name":"agent__explore","arguments":arguments}}),
                    ]
                } else { vec![serde_json::json!({"type":"response.output_text.delta","delta":"done"})] };
                events.push(serde_json::json!({"type":"response.completed","response":{"id":format!("resp-{index}"),"status":"completed"}}));
                let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("letcode.toml");
        let sessions = dir.path().join("sessions");
        fs::write(&config_path, format!(r#"active_provider = "test"
[global]
sessions_dir = "{}"
[subagents]
mode = "swarm"
max_depth = 2
max_descendants = 3
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model.capabilities]
tools = true
"#, toml_path(&sessions))).unwrap();
        let config = AppConfig::load_from_path(&config_path).unwrap();
        let mut recorder = TranscriptRecorder::create(&sessions).unwrap();
        recorder.record_session_started("test/model").unwrap();
        recorder.record_session_title("Root messages").unwrap();
        recorder.record_subagents_policy(&config.subagents).unwrap();
        let root = recorder.session_id().to_string();
        let transcript = Arc::new(StdMutex::new(recorder));
        let factory = Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(config.providers.clone(), config.global.retry.clone(), config.runtime_catalog.clone()));
        let mut agent = Agent::new("model", 3, 6);
        agent.apply_prepared_route(factory.prepare_route(ModelRoute::new("test", "model")).unwrap());
        agent.set_primary_route_factory(factory);
        agent.set_subagents_policy(config.subagents.clone());
        crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
        let mut settings = crate::session_engine_config(&config, Default::default(), String::new(), None);
        settings.route_api_key_configured.insert("test/model".into(), true);
        let runtime = SubagentPool::new();
        let (_mcp, mcp_rx) = mpsc::unbounded_channel();
        let (_reload, reload_rx) = mpsc::unbounded_channel();
        let (control, control_rx) = mpsc::unbounded_channel();
        let (events, mut event_rx) = mpsc::unbounded_channel();
        let (title, _titles) = mpsc::unbounded_channel();
        let task = tokio::spawn(run_engine_loop(agent, transcript.clone(), settings, None, mcp_rx, reload_rx, control_rx, control.clone(), events.clone(), title, runtime.clone(), Default::default(), events));
        control.send(SessionEngineControl::Command(SessionEngineCommand::Prompt(crate::user_content::UserMessageSubmission::new("initial", crate::user_content::UserMessageContent::from("initial"))))).unwrap();
        loop {
            match event_rx.recv().await.unwrap() {
                SessionTransportEvent::Done => break,
                SessionTransportEvent::Error(error) => panic!("{}", error.message),
                _ => {}
            }
        }
        let records = read_records(transcript.lock().unwrap().path()).unwrap();
        runtime.enqueue_member_message(&sessions, &records, &root, "idle member message").unwrap();
        loop {
            match event_rx.recv().await.unwrap() {
                SessionTransportEvent::Done => break,
                SessionTransportEvent::Error(error) => panic!("{}", error.message),
                _ => {}
            }
        }
        let records = read_records(transcript.lock().unwrap().path()).unwrap();
        let users: Vec<_> = records.iter().filter_map(|record| match &record.event { TranscriptEvent::UserMessage { content } => Some(content.text.as_str()), _ => None }).collect();
        let delivered = format!("[Swarm message from session_id={root} to session_id={root}]\n\nidle member message");
        assert_eq!(users, ["initial", delivered.as_str()]);
        let turns: Vec<_> = records.iter().filter_map(|record| match &record.event { TranscriptEvent::TurnStarted(event) => Some(event.turn_id), _ => None }).collect();
        assert_eq!(turns, [1, 2]);
        let mut child = TranscriptRecorder::create(crate::transcript::child_sessions_dir(&sessions)).unwrap();
        child.record_session_started("test/model").unwrap();
        child.record_subagents_policy(&config.subagents).unwrap();
        let member = crate::subagent::SwarmMember { root_session_id: root.clone(), parent_session_id: root.clone(), child_session_id: child.session_id().into(), depth: 1, agent_name: "explorer".into(), pool_ordinal: 1 };
        {
            let mut parent = transcript.lock().unwrap();
            for recorder in [&mut *parent, &mut child] {
                recorder.record_subagent_started("recorded-child", &root, "turn-1", &member.child_session_id, "explorer", "done", 1).unwrap();
            }
            child.record_subagent_activation("recorded-child".into(), member.clone()).unwrap();
            child.record_subagent_lifecycle("recorded-child", &root, "turn-1", "explorer", "completed", None).unwrap();
            parent.record_subagent_result("recorded-child", &root, "turn-1", &member.child_session_id, "explorer", "completed", "done").unwrap();
        }
        drop(child);
        let records = read_records(transcript.lock().unwrap().path()).unwrap();
        assert_eq!(runtime.swarm_snapshot(&sessions, &records).unwrap().members.len(), 1);
        control.send(SessionEngineControl::Command(SessionEngineCommand::NewSession)).unwrap();
        loop { match event_rx.recv().await.unwrap() {
            SessionTransportEvent::SessionStarted { session_id, .. } => { assert_ne!(session_id, root); break; }
            SessionTransportEvent::Error(error) => panic!("{}", error.message), _ => {}
        } }
        assert!(runtime.member_scope_stopped(&root));
        assert!(runtime.enqueue_member_message(&sessions, &records, &root, "blocked").is_err());
        for message in ["after explicit resume", "after self resume"] {
            let previous_generation = runtime.swarm_generation(&root).unwrap();
            control.send(SessionEngineControl::Command(SessionEngineCommand::ResumeSession(root.clone()))).unwrap();
            loop { match event_rx.recv().await.unwrap() {
                SessionTransportEvent::SessionResumed { session_id, .. } => { assert_eq!(session_id, root); break; }
                SessionTransportEvent::Error(error) => panic!("{}", error.message), _ => {}
            } }
            assert!(!runtime.member_scope_stopped(&root));
            assert!(runtime.swarm_generation(&root).unwrap() > previous_generation);
            control.send(SessionEngineControl::Command(SessionEngineCommand::SwarmMessageReady { root_session_id: root.clone(), generation: previous_generation })).unwrap();
            for result in [Ok(run_summary("recorded-child", &member.child_session_id, "explorer", SubagentStatus::Completed)), Err("late failure".into())] {
                control.send(SessionEngineControl::Command(SessionEngineCommand::BackgroundSubagentCompleted { generation: Some(previous_generation), parent_session_id: root.clone(), parent_tool_call_id: None, result })).unwrap();
            }
            let records = read_records(transcript.lock().unwrap().path()).unwrap();
            runtime.enqueue_member_message(&sessions, &records, &root, message).unwrap();
            loop { match event_rx.recv().await.unwrap() {
                SessionTransportEvent::Done => break,
                SessionTransportEvent::Error(error) => panic!("{}", error.message), _ => {}
            } }
            let records = read_records(transcript.lock().unwrap().path()).unwrap();
            let snapshot = runtime.swarm_snapshot(&sessions, &records).unwrap();
            assert_eq!(snapshot.policy, config.subagents);
            assert_eq!(snapshot.members.len(), 2);
            assert_eq!(snapshot.members[&member.child_session_id], member);
        }
        control.send(SessionEngineControl::Shutdown).unwrap();
        task.await.unwrap();
        assert!(runtime.enqueue_member_message(&sessions, &records, &root, "late").unwrap_err().to_string().contains("stopping"));
        assert!(!runtime.is_running());
        let (notifications, _notification_rx) = mpsc::unbounded_channel();
        let mut deliveries = HashMap::new();
        for result in [Ok(run_summary("late-run", "late-child", "explorer", SubagentStatus::Completed)), Err("late failure".into())] {
            assert!(!observe_background_subagent_completion(&transcript, &runtime, &mut deliveries, &notifications, &root, None, &result));
        }
        server.await.unwrap();
    }).await.unwrap();
}
