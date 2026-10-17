use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn test_agent() -> Agent {
    let mut agent = Agent::new("test-model", 1, 1);
    let route = crate::model_runtime::RuntimeConfig::from_toml(
        r#"active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "test-model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models."test-model"]
[providers.test.models."test-model".capabilities]
generation = { max_output_tokens = true }
[providers.test.models."test-model".generation]
max_output_tokens = 128
"#,
    )
    .unwrap()
    .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
    .unwrap()
    .route("test", "test-model")
    .unwrap()
    .clone();
    agent.set_resolved_model_route(Some(Arc::new(route)));
    agent
}

fn protocol_frames_with_spans(
    history: &[HistoryItem],
) -> Vec<crate::protocol_frames::ProtocolFrame> {
    crate::protocol_frames::analyze_history_items(history, None)
        .expect("history is valid")
        .frames
        .into_iter()
        .enumerate()
        .map(|(index, mut frame)| {
            let sequence = index as u64 + 1;
            frame.source_provenance = Some(
                crate::runtime_context::RuntimeFrameProvenance::new(
                    crate::runtime_context::RuntimeSource::Transcript,
                )
                .with_source_id(format!("raw:{sequence}"))
                .with_span(
                    crate::runtime_context::SourceSpan::new(sequence, sequence)
                        .expect("singleton source span"),
                ),
            );
            frame
        })
        .collect()
}

fn prepared(agent: &Agent, history: Vec<HistoryItem>) -> PreparedCompaction {
    let transcript = crate::protocol_frames::analyze_history_items(&history, None)
        .expect("candidate history is valid");
    let mut runtime_snapshot = agent
        .rebuilt_runtime_snapshot_from_protocol_frames(
            &transcript.frames,
            agent.active_protocol_frames().len(),
            &agent.active_history_items(),
        )
        .expect("candidate snapshot");
    merge_non_protocol_runtime_metadata(&mut runtime_snapshot, &agent.runtime_snapshot);
    rebind_active_protocol_from_history(&mut runtime_snapshot, &history)
        .expect("candidate protocol binding");
    runtime_snapshot
        .heal_references()
        .expect("candidate references");
    PreparedCompaction {
        event: ContextCompactionEvent::succeeded_at("summary", Some("raw:2".into())),
        local_state: Some(PreparedLocalCompaction {
            current_turn_start_index: None,
            runtime_snapshot,
        }),
    }
}

#[test]
fn summary_budget_fit_retires_only_the_visible_prefix() {
    let mut agent = test_agent();
    agent.set_default_protocol(ApiProtocol::Responses);
    agent.set_model_catalog(std::collections::HashMap::from([(
        "test-model".into(),
        ModelRequestMetadata {
            context_window: Some(12_000),
            effective_input_limit_tokens: Some(8_000),
            max_output_tokens: Some(128),
            ..ModelRequestMetadata::default()
        },
    )]));
    let history = vec![
        HistoryItem::user("first ".repeat(1_000)),
        HistoryItem::assistant("second ".repeat(1_000)),
        HistoryItem::user("third ".repeat(1_000)),
        HistoryItem::assistant("fourth ".repeat(1_000)),
    ];
    agent
        .replace_history(history.clone())
        .expect("history is protocol-valid");
    let transcript =
        crate::protocol_frames::analyze_history_items(&history, None).expect("history analysis");
    let planned = super::super::history_compact::plan_turn_cut_with_transcript(
        &history,
        None,
        0,
        &transcript,
    )
    .expect("cut planning")
    .expect("history is compactable");

    let fitted = fit_compaction_cut_to_summary_budget(&agent, &history, &transcript, &planned)
        .expect("budget fitting")
        .expect("a smaller prefix fits");

    assert!(fitted.cut_end < planned.cut_end);
    assert_eq!(fitted.prefix, history[..fitted.cut_end]);
    let rendered = render_compaction_history(&fitted.prefix);
    assert!(rendered.contains("first"));
    assert!(!rendered.contains("fourth"));
    assert_eq!(
        super::super::history_compact::compose_with_summary("summary", &history, fitted.cut_end,)
            .expect("compose"),
        std::iter::once(HistoryItem::context_summary("summary"))
            .chain(history[fitted.cut_end..].iter().cloned())
            .collect::<Vec<_>>()
    );
}

#[test]
fn summary_budget_fit_refuses_to_truncate_previous_checkpoint() {
    let mut agent = test_agent();
    agent.set_default_protocol(ApiProtocol::Responses);
    agent.set_model_catalog(std::collections::HashMap::from([(
        "test-model".into(),
        ModelRequestMetadata {
            context_window: Some(4_096),
            effective_input_limit_tokens: Some(2_000),
            max_output_tokens: Some(128),
            ..ModelRequestMetadata::default()
        },
    )]));
    let history = vec![
        HistoryItem::context_summary("checkpoint ".repeat(2_000)),
        HistoryItem::user("new history"),
    ];
    agent
        .replace_history(history.clone())
        .expect("history is protocol-valid");
    let transcript =
        crate::protocol_frames::analyze_history_items(&history, None).expect("history analysis");
    let planned = super::super::history_compact::plan_turn_cut_with_transcript(
        &history,
        None,
        0,
        &transcript,
    )
    .expect("cut planning")
    .expect("new history is compactable");

    let fitted = fit_compaction_cut_to_summary_budget(&agent, &history, &transcript, &planned)
        .expect("budget fitting does not technically fail");

    assert!(fitted.is_none());
}

#[test]
fn bounded_take_chars_stops_at_cap_on_char_boundary() {
    let short = "abc";
    assert_eq!(bounded_take_chars(short, 5), short);
    assert_eq!(bounded_take_chars(short, 3), short);

    // 3 字节的雪花字符，截断必须落在 char 边界（O(cap)，不整串扫描）。
    let long = "ab☃cdef";
    assert_eq!(bounded_take_chars(long, 3), "ab☃");
    assert_eq!(bounded_take_chars(long, 4), "ab☃c");
    assert_eq!(bounded_take_chars(long, 100), long);
}

#[test]
fn render_tool_output_large_json_is_bounded_to_cap_with_marker() {
    // 大 JSON：远超解析前缀，但最终仍被压缩到 ~2000 字符并带截断标记。
    let big_field = "line ".repeat(TOOL_OUTPUT_PARSE_PREFIX_CAP);
    let output = serde_json::json!({ "content": big_field }).to_string();
    assert!(output.chars().count() > TOOL_OUTPUT_PARSE_PREFIX_CAP);

    let rendered = render_tool_output_for_compaction(&output);
    assert!(rendered.chars().count() <= COMPACTION_TOOL_OUTPUT_CHAR_CAP);
    assert!(rendered.contains(COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER));
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn summary_retry_commits_only_successful_output_and_preserves_history_on_failure() {
        for (final_failure, reject_reset) in [(false, false), (true, false), (false, true)] {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                for attempt in 0..if reject_reset { 1 } else { 2 } {
                    let (mut stream, _) =
                        tokio::time::timeout(Duration::from_secs(10), listener.accept())
                            .await
                            .unwrap()
                            .unwrap();
                    let mut request = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let (header_end, content_length) = loop {
                        let read = stream.read(&mut chunk).await.unwrap();
                        assert!(read > 0, "request closed before headers");
                        request.extend_from_slice(&chunk[..read]);
                        let Some(header_end) =
                            request.windows(4).position(|part| part == b"\r\n\r\n")
                        else {
                            continue;
                        };
                        let header_end = header_end + 4;
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
                        let read = stream.read(&mut chunk).await.unwrap();
                        assert!(read > 0, "request closed before body");
                        request.extend_from_slice(&chunk[..read]);
                    }

                    let body = if attempt == 0 || final_failure {
                        concat!(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"failed attempt\"}\n\n",
                            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"retry\"}}}\n\n"
                        )
                    } else {
                        concat!(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"successful summary\"}\n\n",
                            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                        )
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                }
                if reject_reset {
                    assert!(
                        tokio::time::timeout(Duration::from_millis(100), listener.accept())
                            .await
                            .is_err(),
                        "rejected preview reset must not issue another request"
                    );
                }
            });

            let config = crate::model_runtime::RuntimeConfig::from_toml(&format!(
                r#"active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "test-model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.retry]
enabled = true
max_attempts = 2
max_recovery_attempts = 1
initial_delay_secs = 1
exponential_backoff = false
backoff_multiplier = 1.0
jitter_secs = 0
[providers.test.models."test-model"]
[providers.test.models."test-model".capabilities]
generation = {{ max_output_tokens = true }}
[providers.test.models."test-model".generation]
max_output_tokens = 128
"#
            ))
            .unwrap()
            .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
            .unwrap();
            let mut route = config.route("test", "test-model").unwrap().clone();
            route.retry.as_mut().unwrap().initial_delay_secs = 0;
            let mut agent = test_agent();
            agent.set_resolved_model_route(Some(Arc::new(route)));
            let original = vec![
                HistoryItem::user("history"),
                HistoryItem::assistant("reply"),
            ];
            agent.replace_history(original.clone()).unwrap();

            let events = Arc::new(std::sync::Mutex::new(Vec::new()));
            let observed = events.clone();
            let mut starts = 0;
            let mut on_event = move |event| {
                if matches!(event, AgentEvent::ContextCompactionStarted { .. }) {
                    starts += 1;
                    if reject_reset && starts == 2 {
                        return Box::pin(std::future::ready(Err(anyhow!("preview reset failed"))))
                            as BoxFuture<'_, Result<()>>;
                    }
                }
                observed.lock().unwrap().push(event);
                Box::pin(std::future::ready(Ok(()))) as BoxFuture<'_, Result<()>>
            };
            let result = tokio::time::timeout(
                Duration::from_secs(10),
                compact_session_stream_async(&mut agent, &mut on_event, || Ok(())),
            )
            .await
            .expect("local Responses retry test timed out");

            {
                let events = events.lock().unwrap();
                if final_failure || reject_reset {
                    assert!(result.is_err());
                    assert_eq!(agent.history_for_test(), original);
                    assert!(
                        !events
                            .iter()
                            .any(|event| matches!(event, AgentEvent::ContextCompacted(_)))
                    );
                    assert!(matches!(
                        events.last(),
                        Some(AgentEvent::ContextCompactionFailed { .. })
                    ));
                } else {
                    result.expect("summary succeeds after retry");
                    assert_eq!(
                        agent.history_for_test(),
                        vec![HistoryItem::context_summary("successful summary")]
                    );
                    assert!(matches!(
                        events.as_slice(),
                        [
                            AgentEvent::ContextCompactionStarted { trigger: CompactionTrigger::Manual },
                            AgentEvent::ContextCompactionDelta { delta },
                            AgentEvent::ContextCompactionStarted { trigger: CompactionTrigger::Manual },
                            AgentEvent::ContextCompactionDelta { delta: second },
                            AgentEvent::ContextCompacted(committed),
                        ] if delta == "failed attempt" && second == "successful summary" && committed.summary == "successful summary"
                    ));
                }
            }
            tokio::time::timeout(Duration::from_secs(1), server)
                .await
                .expect("local Responses server timed out")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn rejected_durable_compaction_does_not_change_live_history() {
        let mut agent = test_agent();
        let original = vec![HistoryItem::user("old"), HistoryItem::assistant("reply")];
        agent
            .replace_history(original.clone())
            .expect("seed history");
        let candidate = prepared(
            &agent,
            vec![
                HistoryItem::context_summary("summary"),
                HistoryItem::assistant("reply"),
            ],
        );
        let mut on_event = |_event| {
            Box::pin(async { Err(anyhow::anyhow!("durable append rejected")) })
                as BoxFuture<'_, Result<()>>
        };

        assert!(
            commit_prepared_compaction(&mut agent, candidate, &mut on_event)
                .await
                .is_err()
        );
        assert_eq!(agent.history_for_test(), original);
    }

    #[tokio::test]
    async fn acknowledged_provider_compaction_reloads_the_canonical_projection() {
        let mut agent = test_agent();
        agent
            .replace_history(vec![
                HistoryItem::user("old"),
                HistoryItem::assistant("reply"),
            ])
            .expect("seed history");
        let expected = vec![HistoryItem::context_summary("persisted summary")];
        let mut projected = RuntimeSnapshot::new("main");
        projected.workflow.todos = vec![TodoItem {
            id: "durable-todo".into(),
            content: "survives provider compaction reload".into(),
            status: TodoStatus::InProgress,
        }];
        projected.frames = protocol_frames_with_spans(&expected)
            .into_iter()
            .enumerate()
            .map(|(index, frame)| {
                let provenance = frame.source_provenance.expect("source provenance");
                RuntimeFrame::new(
                    RuntimeFrameKind::Summary,
                    FrameVisibility::Active,
                    provenance.clone(),
                    RuntimeFrameIdSeed {
                        frame_kind: RuntimeFrameKind::Summary,
                        source: RuntimeSource::Transcript,
                        ordinal: index as u32,
                        stable_key: provenance.source_id.as_deref().expect("stable source id"),
                        source_span: provenance.source_span,
                    },
                )
                .with_protocol(frame.item)
            })
            .collect();
        agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));
        let candidate = prepared(
            &agent,
            vec![HistoryItem::context_summary("local candidate")],
        );
        let mut on_event = |_event| Box::pin(async { Ok(()) }) as BoxFuture<'_, Result<()>>;

        commit_prepared_compaction(&mut agent, candidate, &mut on_event)
            .await
            .expect("durable acknowledgement reloads provider projection");

        assert_eq!(agent.history_for_test(), expected);
        assert_eq!(agent.runtime_snapshot.workflow.todos.len(), 1);
        assert_eq!(agent.runtime_snapshot.workflow.todos[0].id, "durable-todo");
    }

    #[tokio::test]
    async fn acknowledged_durable_compaction_installs_exact_candidate() {
        let mut agent = test_agent();
        agent
            .replace_history(vec![
                HistoryItem::user("old"),
                HistoryItem::assistant("reply"),
            ])
            .expect("seed history");
        let expected = vec![
            HistoryItem::context_summary("summary"),
            HistoryItem::assistant("reply"),
        ];
        let candidate = prepared(&agent, expected.clone());
        let acknowledged = Arc::new(AtomicBool::new(false));
        let observed = acknowledged.clone();
        let mut on_event = move |event| {
            observed.store(
                matches!(event, AgentEvent::ContextCompacted(_)),
                Ordering::SeqCst,
            );
            Box::pin(async { Ok(()) }) as BoxFuture<'_, Result<()>>
        };

        commit_prepared_compaction(&mut agent, candidate, &mut on_event)
            .await
            .expect("durable acknowledgement installs candidate");

        assert!(acknowledged.load(Ordering::SeqCst));
        assert_eq!(agent.history_for_test(), expected);
    }
}
