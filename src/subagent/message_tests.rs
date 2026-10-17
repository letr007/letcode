#[tokio::test]
async fn member_fifo_starts_distinct_user_turns_at_persisted_boundaries() {
    use crate::model_runtime::{ProtocolRegistry, RuntimeConfig};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::time::{Duration, timeout};
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Stage {
        Complete,
        UserRecord,
        TurnClose,
    }
    struct MessageJournalFailure {
        file: std::fs::File,
        stage: Stage,
        user_attempts: Arc<std::sync::atomic::AtomicUsize>,
        rejected: Arc<AtomicBool>,
    }
    impl JournalSink for MessageJournalFailure {
        fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            let text = String::from_utf8_lossy(bytes);
            let next_user = text.contains("user_message") && text.contains("一二");
            if next_user {
                self.user_attempts.fetch_add(1, Ordering::SeqCst);
            }
            if (self.stage == Stage::UserRecord && next_user)
                || (self.stage == Stage::TurnClose && text.contains("turn_finalized"))
            {
                self.rejected.store(true, Ordering::SeqCst);
                return Err(io::Error::other("message journal write rejected"));
            }
            self.file.write_all(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }
        fn sync_data(&mut self) -> io::Result<()> {
            self.file.sync_data()
        }
    }
    for (root_target, sender_backed, stage) in [
        (false, false, Stage::Complete),
        (false, true, Stage::Complete),
        (true, false, Stage::Complete),
        (true, true, Stage::Complete),
        (false, false, Stage::UserRecord),
        (false, false, Stage::TurnClose),
        (false, true, Stage::TurnClose),
    ] {
        let io_failure = matches!(stage, Stage::UserRecord | Stage::TurnClose);
        let user_attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rejected = Arc::new(AtomicBool::new(false));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (first_tx, first_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            timeout(Duration::from_secs(5), async move {
                let mut first_tx = Some(first_tx);
                let mut release_rx = Some(release_rx);
                for index in 0..if io_failure { 1 } else { 3 } {
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
                    assert!(body["tools"].as_array().unwrap_or_else(|| panic!("root={root_target} sender={sender_backed} index={index} body={body}")).iter().any(|tool| tool["name"] == "agent__send"));
                    if index == 0 {
                        first_tx.take().unwrap().send(()).unwrap();
                        release_rx.take().unwrap().await.unwrap();
                    } else {
                        let input = body["input"].as_array().unwrap();
                        assert!(input.iter().any(|item| item["type"] == "function_call_output"));
                        let user_texts: Vec<_> = input
                            .iter()
                            .filter(|item| item["role"] == "user")
                            .flat_map(|item| item["content"].as_array().unwrap())
                            .filter_map(|part| part["text"].as_str())
                            .collect();
                        assert!(user_texts.iter().any(|text| text.ends_with("\n\n一二")));
                        assert_eq!(
                            user_texts.iter().any(|text| text.ends_with("\n\nsecond")),
                            index == 2
                        );
                    }
                    let events = if index == 0 {
                        let args = json!({"path":"src/subagent.rs"}).to_string();
                        vec![
                            json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-read","call_id":"call-read","name":"fs__read"}}),
                            json!({"type":"response.function_call_arguments.done","item_id":"item-read","arguments":args}),
                            json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-read","call_id":"call-read","name":"fs__read","arguments":args}}),
                        ]
                    } else { vec![json!({"type":"response.output_text.delta","delta":"done"})] };
                    let mut events = events;
                    events.push(json!({"type":"response.completed","response":{"id":format!("response-{index}"),"status":"completed"}}));
                    let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
                }
            }).await.unwrap();
        });
        let catalog = RuntimeConfig::from_toml(&format!(
            r#"active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model.capabilities]
tools = true
parallel_tool_calls = true
[providers.test.models.model.capabilities.generation]
parallel_tool_calls = true
"#
        ))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
        let mut parent = Agent::new("model", 4, 8);
        parent.set_subagents_policy(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_pending_messages: 2,
            max_message_bytes: 8,
            ..Default::default()
        });
        parent.set_model_route_authority(
            ModelRoute::new("test", "model"),
            Arc::new(catalog.route("test", "model").unwrap().clone()),
        );
        parent.set_resolved_runtime_catalog(Some(catalog));
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = TranscriptRecorder::create(dir.path()).unwrap();
        recorder.record_session_started("test/model").unwrap();
        recorder.record_session_title("Message fixture").unwrap();
        recorder
            .record_subagents_policy(parent.subagents_policy())
            .unwrap();
        let root = recorder.session_id().to_string();
        let recorder = Arc::new(Mutex::new(recorder));
        let runtime = SubagentPool::new();
        let (events, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (control, _control_rx) = tokio::sync::mpsc::unbounded_channel();
        let runner = if sender_backed {
            crate::session::runner::AgentRunner::with_transcript(events.clone(), recorder.clone())
        } else {
            crate::session::runner::AgentRunner::silent_with_transcript(recorder.clone())
        }
        .with_subagent_runtime(
            runtime.clone(),
            dir.path().into(),
            Default::default(),
            Default::default(),
            String::new(),
            Some(control),
            sender_backed.then(|| events.clone()),
        );
        runner.install_subagent_delegate_for_test(&mut parent);
        parent.prepare_subagent_runtime().unwrap();
        let (target, path, execution) = if root_target {
            let path = recorder.lock().unwrap().path().to_path_buf();
            let root_copy = root.clone();
            let execution = tokio::spawn(async move {
                runner
                    .run_prompt(
                        &mut parent,
                        crate::user_content::UserMessageSubmission::new(
                            "initial",
                            crate::user_content::UserMessageContent::from("initial"),
                        ),
                    )
                    .await
            });
            (root_copy, path, execution)
        } else {
            let input = test_governance().input;
            let started = runtime
                .start_named_governed(
                    &parent,
                    "explorer",
                    crate::agent::SubagentInvocation {
                        prompt: "initial".into(),
                        input,
                        model: None,
                        parent_tool_call_id: None,
                    },
                    dir.path(),
                    root.clone(),
                    "turn-1".into(),
                    Some(recorder.clone()),
                    sender_backed.then(|| crate::session::runner::subagent_event_sender(events)),
                )
                .unwrap();
            let target = started.receipt().child_session_id.clone();
            let path = crate::transcript::child_session_path(dir.path(), &target);
            let runtime = runtime.clone();
            let user_attempts = user_attempts.clone();
            let rejected = rejected.clone();
            (
                target,
                path,
                tokio::spawn(async move {
                    runtime
                        .complete_started_run_with_executor(
                            started,
                            move |agent, prompt, transcript, sender, child, role| {
                                async move {
                                    if io_failure {
                                        let mut recorder = transcript.lock().unwrap();
                                        let file = std::fs::OpenOptions::new()
                                            .append(true)
                                            .open(recorder.path())
                                            .unwrap();
                                        recorder.replace_sink_for_test(Box::new(
                                            MessageJournalFailure {
                                                file,
                                                stage,
                                                user_attempts,
                                                rejected,
                                            },
                                        ));
                                    }
                                    crate::subagent_events::run_child_prompt(
                                        agent,
                                        prompt,
                                        transcript,
                                        sender,
                                        child,
                                        Some(role),
                                    )
                                    .await
                                }
                                .boxed()
                            },
                        )
                        .await
                        .map(|result| {
                            assert_eq!(
                                result.status,
                                if io_failure {
                                    SubagentStatus::Failed
                                } else {
                                    SubagentStatus::Completed
                                },
                                "{}",
                                result.summary
                            );
                            result.summary
                        })
                }),
            )
        };
        timeout(Duration::from_secs(5), first_rx)
            .await
            .unwrap()
            .unwrap();
        let root_records = read_records(recorder.lock().unwrap().path()).unwrap();
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &target, "一二")
                .unwrap(),
            1
        );
        assert!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &target, "一二三")
                .unwrap_err()
                .to_string()
                .contains("max_message_bytes")
        );
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &target, "second")
                .unwrap(),
            2
        );
        assert!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &target, "third")
                .unwrap_err()
                .to_string()
                .contains("full")
        );
        assert!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, "other-root", "x")
                .is_err()
        );
        let before = read_records(&path).unwrap();
        assert!(!before.iter().any(|record| matches!(
            &record.event,
            crate::transcript::TranscriptEvent::UserMessage { .. }
        )
            && serde_json::to_string(record).unwrap().contains("一二")));
        release_tx.send(()).unwrap();
        timeout(Duration::from_secs(5), execution)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        server.await.unwrap();
        let records = read_records(&path).unwrap();
        if !root_target {
            assert_eq!(
                records
                    .iter()
                    .filter(|record| matches!(
                        &record.event,
                        crate::transcript::TranscriptEvent::SubagentActivated { .. }
                    ))
                    .count(),
                1
            );
        }
        let users: Vec<_> = records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::UserMessage { content, .. } => {
                    Some(content.text.clone())
                }
                _ => None,
            })
            .collect();
        if io_failure {
            assert_eq!(users, ["initial"]);
            assert!(rejected.load(Ordering::SeqCst));
            assert_eq!(
                user_attempts.load(Ordering::SeqCst),
                usize::from(stage == Stage::UserRecord)
            );
            let parent_records = read_records(recorder.lock().unwrap().path()).unwrap();
            let failed_pending = if stage == Stage::TurnClose {
                "2 queued message(s) failed"
            } else {
                "1 queued message(s) failed"
            };
            assert!(parent_records.iter().any(|record| matches!(&record.event, crate::transcript::TranscriptEvent::SubagentResult { summary, .. } if summary.contains(failed_pending))));
        } else {
            let header = format!("[Swarm message from session_id={root} to session_id={target}]");
            assert_eq!(
                users,
                [
                    "initial".to_string(),
                    format!("{header}\n\n一二"),
                    format!("{header}\n\nsecond")
                ]
            );
        }
        let turns: Vec<_> = records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
                _ => None,
            })
            .collect();
        assert_eq!(turns, if io_failure { vec![1] } else { vec![1, 2, 3] });
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(
                    &record.event,
                    crate::transcript::TranscriptEvent::TurnFinalized(_)
                ))
                .count(),
            match stage {
                Stage::TurnClose => 0,
                Stage::UserRecord => 1,
                Stage::Complete => 3,
            }
        );
        assert!(!runtime.member_message_pending(&root, &target).unwrap());
        if sender_backed {
            let mut displayed = Vec::new();
            while let Ok(event) = event_rx.try_recv() {
                match event {
                    SessionTransportEvent::UserMessage(event) => displayed.push(event.content.text),
                    SessionTransportEvent::ChildSessionEvent {
                        child_session_id,
                        event: crate::session::SessionEvent::UserMessage(event),
                        ..
                    } if child_session_id == target => displayed.push(event.content.text),
                    _ => {}
                }
            }
            assert_eq!(displayed, users);
        }
        runtime.cancel_active();
    }
}

#[tokio::test]
async fn peer_messages_resume_one_original_member_with_its_own_authority() {
    for sender_backed in [false, true] {
        let write_target = format!("src/.swarm-permission-{}", generate_run_id());
        let attempted_write = write_target.clone();
        let (parent_finished_tx, parent_finished_rx) = oneshot::channel();
        let (descendant_started_tx, descendant_started_rx) = oneshot::channel();
        use crate::model_runtime::{ProtocolRegistry, RuntimeConfig};
        use serde_json::json;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{Duration, timeout};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            timeout(Duration::from_secs(5), async move {
            let mut parent_finished_tx = Some(parent_finished_tx);
            let mut descendant_started_tx = Some(descendant_started_tx);
            let mut descendant_response = None;
            for index in 0..4 {
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
                assert_eq!(body["model"], if index == 2 { "root-model" } else { "target-model" });
                assert_eq!(body["tools"].as_array().unwrap().iter().any(|tool| tool["name"] == "fs__write"), index != 2);
                let input = body["input"].to_string();
                if index < 2 {
                    assert!(input.contains("original processed task"));
                    assert!(body.to_string().contains("retained compacted context"));
                    assert!(!body.to_string().contains("retired private history"));
                    assert!(input.contains("first"));
                    assert_eq!(input.contains("second"), index == 1);
                } else if index == 2 {
                    assert!(input.contains("wake descendant"));
                } else {
                    assert!(input.contains("original processed task"));
                    assert!(input.contains("first"));
                    assert!(input.contains("second"));
                    assert!(input.contains("third"));
                }
                let mut events = if index == 0 {
                    let arguments = json!({"command":format!("touch {attempted_write}"), "timeout_secs":1}).to_string();
                    vec![
                        json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-write","call_id":"call-write","name":"shell__exec"}}),
                        json!({"type":"response.function_call_arguments.done","item_id":"item-write","arguments":arguments}),
                        json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-write","call_id":"call-write","name":"shell__exec","arguments":arguments}}),
                    ]
                } else { vec![json!({"type":"response.output_text.delta","delta":"done"})] };
                events.push(json!({"type":"response.completed","response":{"id":format!("resp-{index}"),"status":"completed"}}));
                let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                if index == 2 {
                    descendant_response = Some((socket, response));
                    descendant_started_tx.take().unwrap().send(()).unwrap();
                    continue;
                }
                socket.write_all(response.as_bytes()).await.unwrap();
                if index == 1 { parent_finished_tx.take().unwrap().send(()).unwrap(); }
                if index == 3 {
                    let (mut socket, response) = descendant_response.take().unwrap();
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
            }
        }).await.unwrap();
        });
        let catalog = RuntimeConfig::from_toml(&format!(
            r#"active_provider = "root"
[providers.root]
protocol = "responses"
default_model = "root-model"
[providers.root.auth]
type = "none"
[providers.root.endpoints]
base_url = "http://{address}"
[providers.root.models.root-model.capabilities]
tools = true
[providers.target]
protocol = "responses"
default_model = "target-model"
[providers.target.auth]
type = "none"
[providers.target.endpoints]
base_url = "http://{address}"
[providers.target.models.target-model.capabilities]
tools = true
"#
        ))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
        let mut root_agent = Agent::new("root-model", 3, 6);
        root_agent.set_model_route_authority(
            ModelRoute::new("root", "root-model"),
            Arc::new(catalog.route("root", "root-model").unwrap().clone()),
        );
        root_agent.set_resolved_runtime_catalog(Some(catalog.clone()));
        root_agent.set_subagents_policy(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_concurrent: 3,
            ..Default::default()
        });
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [
                    (
                        "general".into(),
                        Some(ModelRoute::new("target", "target-model")),
                        Vec::new(),
                        None,
                    ),
                    (
                        "explorer".into(),
                        Some(ModelRoute::new("root", "root-model")),
                        Vec::new(),
                        None,
                    ),
                ],
                &indexmap::IndexMap::from([
                    (
                        "root".into(),
                        test_provider(
                            &format!("http://{address}"),
                            "key",
                            ApiProtocol::Responses,
                            &["root-model"],
                        ),
                    ),
                    (
                        "target".into(),
                        test_provider(
                            &format!("http://{address}"),
                            "key",
                            ApiProtocol::Responses,
                            &["target-model"],
                        ),
                    ),
                ]),
                &RetryConfig::default(),
            )
            .unwrap()
            .with_runtime_catalog(catalog),
        );
        root_agent.set_primary_route_factory(factory.clone());
        root_agent.set_subagent_child_factory(factory);
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = TranscriptRecorder::create(dir.path()).unwrap();
        recorder.record_session_started("root/root-model").unwrap();
        recorder
            .record_subagents_policy(root_agent.subagents_policy())
            .unwrap();
        let root_id = recorder.session_id().to_string();
        let recorder = Arc::new(Mutex::new(recorder));
        let runtime = SubagentPool::new();
        let (control, _controls) = tokio::sync::mpsc::unbounded_channel();
        let (events, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (stop_events, mut stop_rx) = oneshot::channel();
        let permissions = Arc::new(AtomicBool::new(false));
        let observed_permissions = permissions.clone();
        let event_task = tokio::spawn(async move {
            timeout(Duration::from_secs(5), async move {
            loop { tokio::select! {
                _ = &mut stop_rx => break,
                event = event_rx.recv() => match event {
                    Some(SessionTransportEvent::ChildPermissionRequested { agent_name, handle, .. }) => {
                        assert_eq!(agent_name.as_deref(), Some("general"));
                        observed_permissions.store(true, Ordering::SeqCst);
                        handle.deny().unwrap();
                    }
                    None => break,
                    _ => {}
                }
            } }
        }).await.unwrap();
        });
        let runner = if sender_backed {
            crate::session::runner::AgentRunner::with_transcript(events.clone(), recorder.clone())
        } else {
            crate::session::runner::AgentRunner::silent_with_transcript(recorder.clone())
        };
        runner
            .with_subagent_runtime(
                runtime.clone(),
                dir.path().into(),
                Default::default(),
                Default::default(),
                String::new(),
                Some(control),
                sender_backed.then(|| events.clone()),
            )
            .install_subagent_delegate_for_test(&mut root_agent);
        root_agent.prepare_subagent_runtime().unwrap();
        let mut input = test_governance().input;
        input.owned_paths = vec!["src".into()];
        input.allowed_paths = vec!["Cargo.toml".into()];
        input.forbidden_paths = vec!["src/subagent".into()];
        let started = runtime
            .start_named_governed(
                &root_agent,
                "general",
                crate::agent::SubagentInvocation {
                    prompt: "original processed task".into(),
                    input,
                    model: None,
                    parent_tool_call_id: None,
                },
                dir.path(),
                root_id.clone(),
                "turn-1".into(),
                Some(recorder.clone()),
                sender_backed.then(|| crate::session::runner::subagent_event_sender(events)),
            )
            .unwrap();
        let target = started.receipt().child_session_id.clone();
        let descendant = Arc::new(Mutex::new(None));
        let child_holder = descendant.clone();
        let child_pool = runtime.clone();
        let sessions_dir = dir.path().to_path_buf();
        let target_session = target.clone();
        runtime
            .complete_started_run_with_executor(
                started,
                move |agent, prompt, transcript, _, _, _| {
                    async move {
                        let mut input = test_governance().input;
                        input.allowed_paths = vec!["Cargo.toml".into()];
                        let started = child_pool.start_named_governed(
                            &agent,
                            "explorer",
                            crate::agent::SubagentInvocation {
                                prompt: "active descendant".into(),
                                input,
                                model: None,
                                parent_tool_call_id: None,
                            },
                            &sessions_dir,
                            target_session,
                            "turn-1".into(),
                            Some(transcript.clone()),
                            None,
                        )?;
                        *child_holder.lock().unwrap() = Some(started);
                        let mut recorder = transcript.lock().unwrap();
                        recorder.record_user_message("retired private history")?;
                        recorder.record_assistant_message("retired answer")?;
                        recorder.record_context_compaction(
                            crate::agent::ContextCompactionEvent::succeeded_at(
                                "retained compacted context",
                                None,
                            ),
                        )?;
                        recorder.record_user_message(prompt)?;
                        recorder
                            .record_turn_started(crate::agent::TurnStartedEvent { turn_id: 7 })?;
                        recorder.record_assistant_message("original done")?;
                        recorder.record_turn_finalized(crate::agent::TurnFinalizedEvent {
                            turn_id: 7,
                            outcome: "completed".into(),
                            tool_call_count: 0,
                            continuation_count: 0,
                            write_effects: 0,
                            validation_effects: 0,
                            failed_validation_effects: 0,
                            validation_advisory_emitted: false,
                        })?;
                        Ok("done".into())
                    }
                    .boxed()
                },
            )
            .await
            .unwrap();
        let mut input = test_governance().input;
        input.allowed_paths = vec!["Cargo.toml".into()];
        root_agent.set_permission_mode(crate::permission::PermissionMode::Yolo);
        root_agent.prepare_subagent_runtime().unwrap();
        let peer = runtime
            .start_named_governed(
                &root_agent,
                "explorer",
                crate::agent::SubagentInvocation {
                    prompt: "send to peer".into(),
                    input,
                    model: None,
                    parent_tool_call_id: None,
                },
                dir.path(),
                root_id.clone(),
                "turn-1".into(),
                Some(recorder.clone()),
                None,
            )
            .unwrap();
        let target_copy = target.clone();
        let peer_session_id = peer.receipt().child_session_id.clone();
        runtime
            .complete_started_run_with_executor(peer, move |agent, _, _, _, _, _| {
                async move {
                    let first = json!({"session_id":target_copy,"message":"first"});
                    let second = json!({"session_id":target_copy,"message":"second"});
                    let (first, second) = tokio::join!(
                        agent.execute_subagent_control_tool_for_test("agent__send", &first),
                        agent.execute_subagent_control_tool_for_test("agent__send", &second)
                    );
                    assert!(first.ok, "{first:?}");
                    assert!(second.ok, "{second:?}");
                    assert!(first.data.unwrap()["accepted"].as_bool().unwrap());
                    Ok("sent".into())
                }
                .boxed()
            })
            .await
            .unwrap();
        let descendant = descendant.lock().unwrap().take().unwrap();
        assert!(
            runtime
                .active_jobs()
                .iter()
                .any(|job| job.run_id == descendant.run_id())
        );
        timeout(Duration::from_secs(5), parent_finished_rx)
            .await
            .unwrap()
            .unwrap();
        if let Some(run) = runtime
            .active_jobs()
            .into_iter()
            .find(|job| job.child_session_id == target)
        {
            runtime.wait_for_result(&run.run_id).await.unwrap();
        }
        let descendant_id = descendant.receipt().child_session_id.clone();
        runtime
            .complete_started_run_with_executor(descendant, |_, prompt, transcript, _, _, _| {
                async move {
                    transcript.lock().unwrap().record_user_message(prompt)?;
                    Ok("descendant completed".into())
                }
                .boxed()
            })
            .await
            .unwrap();
        assert!(
            TranscriptRecorder::open(crate::transcript::child_sessions_dir(dir.path()), &target,)
                .is_ok()
        );
        let root_records = read_records(recorder.lock().unwrap().path()).unwrap();
        assert_eq!(
            runtime
                .enqueue_member_message(
                    dir.path(),
                    &root_records,
                    &descendant_id,
                    "wake descendant"
                )
                .unwrap(),
            1
        );
        timeout(Duration::from_secs(5), descendant_started_rx)
            .await
            .unwrap()
            .unwrap();
        for session_id in [&target, &descendant_id] {
            assert!(
                TranscriptRecorder::open(
                    crate::transcript::child_sessions_dir(dir.path()),
                    session_id,
                )
                .is_err()
            );
        }
        assert_eq!(
            runtime
                .enqueue_member_message(dir.path(), &root_records, &target, "third")
                .unwrap(),
            1
        );
        server.await.unwrap();
        assert!(runtime.wait_until_idle().await);
        for session_id in [&target, &descendant_id] {
            assert!(
                TranscriptRecorder::open(
                    crate::transcript::child_sessions_dir(dir.path()),
                    session_id,
                )
                .is_ok()
            );
        }
        let child_records = read_records(crate::transcript::child_session_path(
            dir.path(),
            &descendant_id,
        ))
        .unwrap();
        let child_members: Vec<_> = child_records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::SubagentActivated { member, .. } => {
                    Some(member)
                }
                _ => None,
            })
            .collect();
        assert_eq!(child_members.len(), 2);
        assert_eq!(child_members[0], child_members[1]);
        assert_eq!(child_members[1].parent_session_id, target);
        let records =
            read_records(crate::transcript::child_session_path(dir.path(), &target)).unwrap();
        let wrote = std::path::Path::new(&write_target).exists();
        if wrote {
            std::fs::remove_file(&write_target).unwrap();
        }
        assert!(!wrote);
        assert_eq!(permissions.load(Ordering::SeqCst), sender_backed);
        assert!(records.iter().any(|record| matches!(&record.event, crate::transcript::TranscriptEvent::PermissionDecision { tool, allowed: false, reason, .. } if tool == "shell__exec" && (sender_backed || reason.as_deref() == Some("Denied in headless child execution")))));
        let activations: Vec<_> = records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::SubagentActivated { member, .. } => {
                    Some(member)
                }
                _ => None,
            })
            .collect();
        assert_eq!(activations.len(), 3);
        assert!(activations.iter().all(|member| *member == activations[0]));
        assert_eq!(activations[1].agent_name, "general");
        assert_eq!(activations[1].parent_session_id, root_id);
        let turns: Vec<_> = records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
                _ => None,
            })
            .collect();
        assert_eq!(turns, [7, 8, 9, 10]);
        let users: Vec<_> = records
            .iter()
            .filter_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::UserMessage { content } => {
                    Some(content.text.as_str())
                }
                _ => None,
            })
            .collect();
        let header =
            format!("[Swarm message from session_id={peer_session_id} to session_id={target}]");
        assert_eq!(
            users,
            [
                "retired private history".to_string(),
                "original processed task".to_string(),
                format!("{header}\n\nfirst"),
                format!("{header}\n\nsecond"),
                format!(
                    "[Swarm message from session_id={root_id} to session_id={target}]\n\nthird"
                )
            ]
        );
        let root_records = read_records(recorder.lock().unwrap().path()).unwrap();
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &root_records)
                .unwrap()
                .members
                .len(),
            3
        );
        assert!(!runtime.member_message_pending(&root_id, &target).unwrap());
        runtime.cancel_active();
        let _ = stop_events.send(());
        event_task.await.unwrap();
    }
}
