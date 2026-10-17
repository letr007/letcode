use super::*;
use crate::transcript::{TranscriptEvent, TranscriptRecorder};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn agent_with_recorder(root: &std::path::Path) -> (Agent, Arc<Mutex<TranscriptRecorder>>) {
    let config = crate::model_runtime::RuntimeConfig::from_toml(
        r#"
active_provider="test"
[providers.test]
protocol="responses"
default_model="m"
[providers.test.auth]
type="none"
[providers.test.endpoints]
base_url="http://127.0.0.1:1"
[providers.test.models.m]
[providers.test.models.m.capabilities]
generation={max_output_tokens=true}
[providers.test.models.m.generation]
max_output_tokens=128
"#,
    )
    .unwrap()
    .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
    .unwrap();
    let mut agent = Agent::new("m", 5, 0);
    agent.set_primary_route(crate::config::ModelRoute::new("test", "m"));
    agent.set_resolved_model_route(Some(Arc::new(config.route("test", "m").unwrap().clone())));
    let mut recorder = TranscriptRecorder::create(root).unwrap();
    recorder.record_session_started("test/m").unwrap();
    let recorder = Arc::new(Mutex::new(recorder));
    let projected = recorder.clone();
    agent.set_runtime_snapshot_provider(Arc::new(move || {
        let recorder = projected.lock().unwrap();
        Ok(
            crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                recorder.session_id().into(),
                crate::transcript::read_records(recorder.path())?,
                crate::transcript::transcript_projection::SessionContextCursor {
                    branch_id: None,
                    leaf_sequence: None,
                },
                &[],
            )?
            .snapshot,
        )
    }));
    agent.reload_runtime_snapshot_from_provider().unwrap();
    (agent, recorder)
}

// Mirrors the live append/persist order, including non-protocol journal rows
// between results. A synthetic live span must never become a raw journal ID.
fn append_live_group(
    agent: &mut Agent,
    recorder: &Arc<Mutex<TranscriptRecorder>>,
    prefix: &str,
    count: usize,
) {
    let calls: Vec<_> = (0..count)
        .map(|i| HistoryToolCall {
            call_id: format!("{prefix}-{i}"),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"source.rs"}"#.into(),
        })
        .collect();
    agent
        .append_history_item(HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: calls.clone(),
        })
        .unwrap();
    recorder
        .lock()
        .unwrap()
        .record_assistant_tool_call_batch(None, None, None, calls.clone())
        .unwrap();
    for call in &calls {
        recorder
            .lock()
            .unwrap()
            .record_tool_call_started(
                &call.call_id,
                &call.name,
                serde_json::json!({"path":"source.rs"}),
            )
            .unwrap();
    }
    for call in &calls {
        let output = crate::tool::ToolResult::ok(
            "fs__read",
            serde_json::json!({"content":format!("unique body {}",call.call_id)}),
        );
        recorder
            .lock()
            .unwrap()
            .record_tool_call_finished(&call.call_id, &call.name, true, output.clone())
            .unwrap();
        agent
            .append_history_item(HistoryItem::ToolOutput {
                call_id: call.call_id.clone(),
                output_json: serde_json::to_string(&output).unwrap(),
                images: vec![],
            })
            .unwrap();
    }
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn a_rejected_publication_is_repaired_and_both_attempts_are_recorded() {
        // Driven to its terminal state by polling, so the test never waits on
        // wall-clock time.
        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                // The first response is rejected and the repair is accepted, so
                // the exchange log carries both outcomes.
                for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0u8; 4096];
                let (header_end, length) = loop {
                    let n = stream.read(&mut buffer).await.unwrap();
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(at) = bytes.windows(4).position(|p| p == b"\r\n\r\n") {
                        let end = at + 4;
                        let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap();
                        break (end, length);
                    }
                };
                while bytes.len() < header_end + length {
                    let n = stream.read(&mut buffer).await.unwrap();
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let publication = if attempt == 0 {
                    "<episode><end>0</end><title>Rejected</title><importance>95</importance><detailed>not covered</detailed><compact>not covered</compact><anchor>rejected</anchor></episode>".to_string()
                } else {
                    // Any chunk has at least one source, so a one-episode repair
                    // is accepted whatever this session holds.
                    "<episode><end>1</end><title>Repaired</title><importance>95</importance><detailed>covered</detailed><compact>covered</compact><anchor>repaired</anchor></episode>".to_string()
                };
                let delta = serde_json::json!({
                    "type": "response.output_text.delta",
                    "delta": publication
                });
                let terminal = serde_json::json!({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 321, "output_tokens": 45, "total_tokens": 366}
                    }
                });
                let body = format!("data: {delta}\n\ndata: {terminal}\n\n");
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                }
            });
            let config = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                r#"active_provider="test"
[providers.test]
protocol="responses"
default_model="test-model"
[providers.test.auth]
type="none"
[providers.test.endpoints]
base_url="http://{address}"
[providers.test.retry]
enabled=false
max_attempts=1
max_recovery_attempts=1
initial_delay_secs=1
exponential_backoff=false
backoff_multiplier=1.0
jitter_secs=0
[providers.test.models."test-model"]
[providers.test.models."test-model".capabilities]
input_images=true
generation={{max_output_tokens=true}}
[providers.test.models."test-model".generation]
max_output_tokens=4096
"#
            ))
            .unwrap()
            .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
            .unwrap();
            let route = config.route("test", "test-model").unwrap().clone();
            let mut agent = Agent::new("test-model", 5, 0);
            agent.set_model_catalog(std::collections::HashMap::from([(
                "test-model".into(),
                ModelRequestMetadata {
                    context_window: Some(32_768),
                    max_output_tokens: Some(4096),
                    ..Default::default()
                },
            )]));
            agent.set_primary_route(crate::config::ModelRoute::new("test", "test-model"));
            agent.set_resolved_model_route(Some(Arc::new(route)));
            let directory = tempfile::tempdir().unwrap();
            let mut recorder = TranscriptRecorder::create(directory.path()).unwrap();
            recorder.record_session_started("test/test-model").unwrap();
            recorder
                .record_user_message_content(UserMessageContent::new("Fix parser", vec![]))
                .unwrap();
            recorder
                .record_assistant_tool_call_batch(
                    None,
                    Some("READABLE-REASONING".into()),
                    None,
                    vec![HistoryToolCall {
                        call_id: "read-1".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"parser.rs"}"#.into(),
                    }],
                )
                .unwrap();
            recorder
                .record_tool_call_finished(
                    "read-1",
                    "fs__read",
                    true,
                    crate::tool::ToolResult::ok(
                        "fs__read",
                        serde_json::json!({"content":"parser detail"}),
                    ),
                )
                .unwrap();
            recorder
                .record_assistant_message("Fixed UTF-8 parser offsets")
                .unwrap();
            let recorder = Arc::new(Mutex::new(recorder));
            let projected = recorder.clone();
            agent.set_runtime_snapshot_provider(Arc::new(move || {
                let recorder = projected.lock().unwrap();
                let records = crate::transcript::read_records(recorder.path())?;
                Ok(crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().to_string(),
                    records,
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?
                .snapshot)
            }));
            agent.reload_runtime_snapshot_from_provider().unwrap();
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let runtime = Arc::new(crate::session::historian::HistorianRuntime::new(
                crate::subagent::SubagentPool::new(),
                directory.path().into(),
                recorder.clone(),
                tx,
            ));
            agent.historian_runtime = Some(runtime.clone());
            let prepared = work(&agent, true, None).unwrap().unwrap();
            let source_count = prepared.source_ids.len();
            runtime.start(&agent, prepared).unwrap();
            let session = agent.runtime_snapshot.session_id.clone().unwrap_or_default();
            let branch = agent.runtime_snapshot.active_context.branch_id.clone();
            let revision = agent.runtime_snapshot.context_scope_revision;
            let mut rounds = 0;
            let published = loop {
                match runtime.poll(&session, &branch, revision) {
                    Ok(None) => {}
                    Ok(Some(publication)) => break publication,
                    Err(error) => panic!("the repaired attempt must publish: {error:#}"),
                }
                rounds += 1;
                assert!(rounds < 100_000, "historian run did not finish");
                tokio::task::yield_now().await;
            };
            assert_eq!(
                published.source_ids.len(),
                1,
                "the repaired response publishes the prefix it covers"
            );
            let records = crate::transcript::read_records(recorder.lock().unwrap().path()).unwrap();
            let child_id = records
                .iter()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentStarted {
                        child_session_id,
                        agent_name,
                        ..
                    } if agent_name == "historian" => Some(child_session_id.clone()),
                    _ => None,
                })
                .unwrap();
            let child_records =
                crate::transcript::read_child_session_records(directory.path(), &child_id).unwrap();
            let exchanges: Vec<_> = child_records
                .iter()
                .filter_map(|record| match &record.event {
                    TranscriptEvent::HistorianExchange(exchange) => Some(exchange.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                exchanges
                    .iter()
                    .map(|exchange| exchange.outcome.as_str())
                    .collect::<Vec<_>>(),
                vec!["rejected", "prepared"],
                "the rejected attempt and the accepted repair are both recorded"
            );
            let exchange = exchanges.first().expect("the rejection is recorded").clone();
            assert_eq!(exchange.source_count, source_count);
            assert!(exchange.payload_bytes > 0);
            assert_eq!(exchange.input_tokens, Some(321));
            assert_eq!(exchange.output_tokens, Some(45));
            assert!(
                exchange
                    .error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("expected a cut after"),
                "unexpected rejection reason: {:?}",
                exchange.error
            );
            let response = exchange.response.unwrap_or_default();
            assert_eq!(
                exchange.response_chars,
                Some(response.chars().count() as u64)
            );
            assert!(response.contains("<end>0</end>"));
            server.await.unwrap();
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn one_historian_work_covers_a_bounded_chunk_of_the_uncovered_region() {
        use crate::request_builder::HistoryToolCall;
        let root = tempfile::tempdir().unwrap();
        let (mut agent, recorder) = agent_with_recorder(root.path());
        {
            let mut recorder = recorder.lock().unwrap();
            recorder
                .record_turn_started(crate::agent::TurnStartedEvent { turn_id: 1 })
                .unwrap();
            recorder.record_user_message("Read every source").unwrap();
            for index in 0..40 {
                let call_id = format!("read-{index}");
                recorder
                    .record_assistant_tool_call_batch(
                        None,
                        None,
                        None,
                        vec![HistoryToolCall {
                            call_id: call_id.clone(),
                            name: "fs__read".into(),
                            arguments_json: r#"{"path":"source.rs"}"#.into(),
                        }],
                    )
                    .unwrap();
                recorder
                    .record_tool_call_finished(
                        &call_id,
                        "fs__read",
                        true,
                        crate::tool::ToolResult::ok(
                            "fs__read",
                            serde_json::json!({"content": "source detail ".repeat(3_000)}),
                        ),
                    )
                    .unwrap();
            }
        }
        agent.turn.current_turn_start_index = Some(0);
        agent.reload_runtime_snapshot_from_provider().unwrap();
        let region = persisted_history_snapshot(&agent)
            .unwrap()
            .active_protocol_frames()
            .len();
        let prepared = work(&agent, true, None).unwrap().unwrap();
        assert!(!prepared.source_ids.is_empty());
        assert!(
            prepared.source_ids.len() < region,
            "one call must not read the whole region: {} of {region}",
            prepared.source_ids.len()
        );
        let oldest = persisted_history_snapshot(&agent)
            .unwrap()
            .active_protocol_frames()[0]
            .source_provenance
            .as_ref()
            .and_then(|provenance| provenance.source_id.clone());
        assert_eq!(Some(prepared.source_ids[0].clone()), oldest);
    }

    #[tokio::test]
    async fn persisted_sources_prepare_and_apply_while_live_appends_keep_request_identity() {
        let root = tempfile::tempdir().unwrap();
        let (mut agent, recorder) = agent_with_recorder(root.path());
        recorder
            .lock()
            .unwrap()
            .record_user_message("Inspect source")
            .unwrap();
        recorder
            .lock()
            .unwrap()
            .record_turn_started(TurnStartedEvent { turn_id: 1 })
            .unwrap();
        agent.turn.current_turn_start_index = Some(0);
        agent
            .append_history_item(HistoryItem::user("Inspect source"))
            .unwrap();
        append_live_group(&mut agent, &recorder, "first", 8);
        assert!(
            agent.active_protocol_frames().iter().all(|f| f
                .source_provenance
                .as_ref()
                .unwrap()
                .source_id
                .is_none())
        );
        let before = agent.runtime_snapshot.clone();
        let frontier = agent.protocol_append_state.frontier_token();
        let generation = agent.protocol_append_state.generation();
        let prepared = work(&agent, true, None).unwrap().unwrap();
        assert_eq!(
            agent.runtime_snapshot, before,
            "background preparation must not install a different runtime"
        );
        assert_eq!(agent.protocol_append_state.frontier_token(), frontier);
        assert_eq!(agent.protocol_append_state.generation(), generation);
        let canonical = persisted_history_snapshot(&agent).unwrap();
        let expected: Vec<_> = canonical
            .active_protocol_frames()
            .iter()
            .map(|f| {
                f.source_provenance
                    .as_ref()
                    .unwrap()
                    .source_id
                    .clone()
                    .unwrap()
            })
            .collect();
        assert_eq!(prepared.source_ids, expected);
        assert_eq!(prepared.source_ids.len(), 10);
        assert!(prepared.input.text.contains("unique body first-7"));
        let mut publication = crate::historian::parse_publication(
            &prepared.id,
            &prepared.source_ids,
            "<episode><end>10</end><title>Inspected sources</title><importance>70</importance><detailed>Detailed result</detailed><compact>Result</compact><anchor>Sources</anchor></episode>",
        )
        .unwrap();
        publication.project_path = Some(prepared.project_path);
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryPublished(publication),
                prepared.revision,
            )
            .unwrap();
        append_live_group(&mut agent, &recorder, "second", 2);
        let canonical = persisted_history_snapshot(&agent).unwrap();
        let tail = canonical.active_protocol_frames()[10..].to_vec();
        let application = pending_application(&agent, true).unwrap().unwrap();
        assert_eq!(
            application.first_kept_entry_id,
            tail[0].source_provenance.as_ref().unwrap().source_id
        );
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryApplied(application),
                prepared.revision,
            )
            .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();
        assert_eq!(
            agent.active_history_items().last(),
            Some(&tail.last().unwrap().item)
        );
        let second = work(&agent, true, None).unwrap().unwrap();
        let publication = crate::historian::parse_publication(
            &second.id,
            &second.source_ids,
            &format!(
                "<episode><end>{}</end><title>Second group</title><importance>70</importance><detailed>Second result</detailed><compact>Result</compact><anchor>Second</anchor></episode>",
                second.source_ids.len()
            ),
        )
        .unwrap();
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryPublished(publication),
                second.revision,
            )
            .unwrap();
        let second_application = pending_application(&agent, true).unwrap().unwrap();
        assert!(second_application.first_kept_entry_id.is_none());
        recorder
            .lock()
            .unwrap()
            .record_history_event(
                TranscriptEvent::HistoryApplied(second_application),
                second.revision,
            )
            .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();
        append_live_group(&mut agent, &recorder, "third", 1);
        assert!(
            agent
                .active_protocol_frames()
                .iter()
                .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
                .all(|f| f.source_provenance.as_ref().unwrap().source_id.is_none())
        );
        let expected_tail: Vec<_> = persisted_history_snapshot(&agent)
            .unwrap()
            .active_protocol_frames()
            .into_iter()
            .filter(|f| !matches!(f.item, HistoryItem::ContextSummary { .. }))
            .map(|f| f.item)
            .collect();
        // Reapplying the frozen selection must retain the persisted raw boundary.
        let application = pending_application(&agent, true).unwrap();
        let observed = recorder.clone();
        let mut events = move |event| {
            std::future::ready(
                crate::agent_event_journal::persist_agent_event(
                    &mut observed.lock().unwrap(),
                    &event,
                )
                .map(|_| ()),
            )
        };
        if let Some(application) = application {
            apply_history(&mut agent, application, false, &mut events)
                .await
                .unwrap();
        }
        let actual_tail: Vec<_> = agent
            .active_history_items()
            .into_iter()
            .filter(|item| !matches!(item, HistoryItem::ContextSummary { .. }))
            .collect();
        assert_eq!(actual_tail, expected_tail);
    }

    #[test]
    fn live_history_matches_the_journal_projection() {
        // The live view may lead the journal, but it must never hold an item the
        // journal cannot express: a restart projects this same history.
        let root = tempfile::tempdir().unwrap();
        let (mut agent, recorder) = agent_with_recorder(root.path());
        recorder
            .lock()
            .unwrap()
            .record_user_message("Inspect source")
            .unwrap();
        agent
            .append_history_item(HistoryItem::user("Inspect source"))
            .unwrap();
        append_live_group(&mut agent, &recorder, "first", 2);

        assert_eq!(
            agent.active_history_items(),
            persisted_history_snapshot(&agent)
                .unwrap()
                .active_history_items()
        );
    }

    #[test]
    fn historian_sources_reject_a_different_persisted_scope() {
        let root = tempfile::tempdir().unwrap();
        let (mut agent, _recorder) = agent_with_recorder(root.path());
        let original = agent.runtime_snapshot.clone();
        for field in ["session", "branch", "revision"] {
            agent.runtime_snapshot = original.clone();
            match field {
                "session" => agent.runtime_snapshot.session_id = None,
                "branch" => agent.runtime_snapshot.active_context.branch_id = "other".into(),
                _ => agent.runtime_snapshot.context_scope_revision += 1,
            }
            let error = work(&agent, true, None)
                .err()
                .expect("scope mismatch")
                .to_string();
            assert!(error.contains("scope changed"), "{field}: {error}");
            assert!(error.contains("active session="), "{error}");
            assert!(error.contains("persisted session="), "{error}");
        }
    }

    #[tokio::test]
    async fn history_application_restates_workflow_state_after_compaction() {
        let root = tempfile::tempdir().unwrap();
        let (mut agent, recorder) = agent_with_recorder(root.path());
        recorder
            .lock()
            .unwrap()
            .record_todo_snapshot(vec![
                TodoItem {
                    id: "open".into(),
                    content: "finish parser".into(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    id: "done".into(),
                    content: "scaffold module".into(),
                    status: TodoStatus::Completed,
                },
            ])
            .unwrap();
        agent.reload_runtime_snapshot_from_provider().unwrap();

        let application = HistoryApplication {
            publication_ids: Vec::new(),
            baseline: Vec::new(),
            delta: Vec::new(),
            baseline_fact_ids: Vec::new(),
            delta_fact_ids: Vec::new(),
            withdrawn_fact_ids: Vec::new(),
            first_kept_entry_id: None,
            legacy_summary: None,
        };
        let mut events = Vec::new();
        {
            let mut on_event = |event: AgentEvent| {
                let result = crate::agent_event_journal::persist_agent_event(
                    &mut recorder.lock().unwrap(),
                    &event,
                )
                .map(|_| ());
                events.push(event);
                std::future::ready(result)
            };
            apply_history(&mut agent, application, false, &mut on_event)
                .await
                .unwrap();
        }

        assert!(matches!(
            events.last(),
            Some(AgentEvent::InternalContinuation {
                source: crate::transcript::InternalContinuationSource::CompactionReminder,
                ..
            })
        ));

        // The reminder is durable, so a restart still shows the open work.
        agent.reload_runtime_snapshot_from_provider().unwrap();
        let history = agent.active_history_items();
        let Some(HistoryItem::InternalContinuation { text }) = history.last() else {
            panic!("expected the workflow reminder to survive the reload");
        };
        assert!(text.contains("finish parser"), "{text}");
        assert!(!text.contains("scaffold module"), "{text}");
    }
}
