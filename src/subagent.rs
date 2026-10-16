mod pool;
mod result;
mod route_factory;
pub(crate) mod swarm;
pub use swarm::SwarmMember;

pub(crate) use pool::generate_run_id as generate_message_id;
pub(crate) use pool::{MessageDelegateFactory, MessageHost, SubagentCapacityExceeded};
pub use pool::{SubagentJob, SubagentPool, SubagentRunGovernance};
pub(crate) use result::REPORT_PROMPT;
pub use result::{
    StructuredSubagentResult, SubagentFailureKind, SubagentRunSummary, SubagentStatus,
    try_parse_structured_subagent_result,
};
pub use route_factory::ExpertRouteFactory;

#[cfg(test)]
use crate::agent::{Agent, AgentFactory, AgentTemplate, PrimaryRouteFactory, SubagentChildFactory};
#[cfg(test)]
use crate::config::{ModelRoute, ProviderConfig, RetryConfig};
#[cfg(test)]
use crate::tool::NormalizedSubagentInput;
#[cfg(test)]
use crate::transcript::TranscriptRecorder;
#[cfg(test)]
use anyhow::Result;
#[cfg(test)]
use futures_util::FutureExt;
#[cfg(test)]
use pool::generate_run_id;
#[cfg(test)]
use result::{build_completed_summary, build_runtime_summary, classify_failure_status};
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests {
    use super::*;
    include!("subagent/message_tests.rs");

    #[test]
    fn text_with_a_closing_brace_before_an_opening_brace_is_not_a_candidate() {
        let text = "任务描述 } 之后还有一个未闭合的 { 左括号";
        assert!(try_parse_structured_subagent_result(text).is_none());
    }

    #[tokio::test]
    async fn cancel_active_covers_starting_reservations() {
        use crate::session::engine::{
            SessionEngineCommand, SessionEngineControl, wait_for_subagent_cancel_settle,
        };
        use std::collections::VecDeque;
        use std::time::Duration;
        let runtime = SubagentPool::new();
        let reservation = runtime.starting_reservation_for_test();
        let run_id = reservation.run_id().to_string();
        assert_eq!(runtime.cancel_active_run_ids(), vec![run_id]);
        let cancel_rx = reservation.activate();
        assert!(cancel_rx.await.is_ok());

        for (settles, requests_shutdown) in
            [(true, false), (true, true), (false, false), (false, true)]
        {
            let runtime = SubagentPool::new();
            let reservation = runtime.starting_reservation_for_test();
            let run_id = reservation.run_id().to_string();
            runtime.cancel_active();
            let (control, mut control_rx) = tokio::sync::mpsc::unbounded_channel();
            control
                .send(SessionEngineControl::Command(
                    SessionEngineCommand::SetModel("queued".into()),
                ))
                .unwrap();
            if requests_shutdown {
                control.send(SessionEngineControl::Shutdown).unwrap();
            }
            let (parent_done, parent_done_rx) = oneshot::channel();
            let (release, release_rx) = oneshot::channel();
            let worker = tokio::spawn(async move {
                parent_done_rx.await.unwrap();
                if !settles {
                    release_rx.await.unwrap();
                }
                drop(reservation);
            });
            let parent = async move {
                parent_done.send(()).unwrap();
            };
            tokio::pin!(parent);
            let mut deferred = VecDeque::new();
            let result = wait_for_subagent_cancel_settle(
                &mut control_rx,
                &mut deferred,
                parent.as_mut(),
                &runtime,
                Duration::from_millis(200),
            )
            .await;
            assert_eq!(result.shutdown, requests_shutdown);
            assert_eq!(result.settled, settles);
            assert_eq!(
                result.remaining_run_ids,
                if settles {
                    Vec::new()
                } else {
                    vec![run_id.clone()]
                }
            );
            assert!(
                matches!(deferred.pop_front(), Some(SessionEngineCommand::SetModel(model)) if model == "queued")
            );
            assert!(deferred.is_empty());
            if !settles {
                assert_eq!(
                    runtime
                        .active_jobs()
                        .into_iter()
                        .map(|job| job.run_id)
                        .collect::<Vec<_>>(),
                    [run_id]
                );
                release.send(()).unwrap();
            }
            worker.await.unwrap();
            assert!(!runtime.is_running());
        }
    }

    use crate::config::ApiProtocol;
    use crate::session::SessionTransportEvent;
    use crate::transcript::{JournalSink, read_records};
    use std::io::{self, Write};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use tokio::sync::{Barrier, oneshot};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn test_agent() -> Agent {
        Agent::new("gpt-test", 2, 4)
    }

    fn temp_sessions_dir() -> PathBuf {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("{}-{id}", generate_run_id()))
    }

    fn install_delegate(
        parent: &mut Agent,
        runtime: &SubagentPool,
        sessions_dir: &std::path::Path,
        recorder: Arc<Mutex<TranscriptRecorder>>,
    ) {
        crate::session::runner::AgentRunner::silent_with_transcript(recorder)
            .with_subagent_runtime(
                runtime.clone(),
                sessions_dir.to_path_buf(),
                Default::default(),
                Default::default(),
                String::new(),
                None,
                None,
            )
            .install_subagent_delegate_for_test(parent);
    }

    fn no_event_sender() -> Option<crate::subagent_events::SubagentEventSender> {
        None
    }

    fn test_governance() -> SubagentRunGovernance {
        SubagentRunGovernance {
            timeout_secs: None,
            max_tool_calls: None,
            model: None,
            input: NormalizedSubagentInput {
                objective: "test".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: None,
                target_child_session_id: None,
                background: false,
            },
        }
    }

    #[tokio::test]
    async fn cancellation_before_executor_poll_does_not_execute_executor() {
        let runtime = SubagentPool::new();
        let started = runtime
            .start_named_governed(
                &test_agent(),
                "explorer",
                crate::agent::SubagentInvocation {
                    prompt: "cancel before execution".into(),
                    input: test_governance().input,
                    model: None,
                    parent_tool_call_id: None,
                },
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
            )
            .expect("run starts");
        let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed_executions = Arc::clone(&executions);
        assert!(!runtime.cancel_active_run_ids().is_empty());

        let summary = runtime
            .complete_started_run_with_executor(
                started,
                |_agent, _task, _transcript, _sender, _child, _name| {
                    async move {
                        observed_executions.fetch_add(1, Ordering::SeqCst);
                        Ok("must not execute".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("cancelled run settles");

        assert_eq!(summary.status, SubagentStatus::Cancelled);
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn started_named_run_exposes_receipt_before_completion() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let mut invocation_input = test_governance().input;
        invocation_input.background = true;
        let invocation = crate::agent::SubagentInvocation {
            prompt: "test".into(),
            input: invocation_input,
            model: None,
            parent_tool_call_id: Some("call-bg".into()),
        };
        let started = runtime
            .start_named_governed(
                &agent,
                "explorer",
                invocation,
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
            )
            .expect("background run starts");

        assert_eq!(started.receipt().agent_name, "explorer");
        assert_eq!(started.receipt().status, "running");
        assert!(runtime.is_running());
        runtime.cancel_active();
        let result = runtime
            .complete_started_run(started)
            .await
            .expect("cancelled run settles");
        assert_eq!(result.status, SubagentStatus::Cancelled);
    }

    #[tokio::test]
    async fn new_child_starts_with_canonical_runtime_snapshot() {
        let runtime = SubagentPool::new();
        let mut parent = test_agent();
        parent.set_subagents_policy(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_depth: 8,
            ..Default::default()
        });
        let expected_policy = parent.subagents_policy().clone();
        let sessions_dir = temp_sessions_dir();
        let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
        recorder.record_session_started("gpt-test").unwrap();
        recorder.record_subagents_policy(&expected_policy).unwrap();
        let parent_id = recorder.session_id().to_string();
        let recorder = Arc::new(Mutex::new(recorder));
        install_delegate(&mut parent, &runtime, &sessions_dir, recorder.clone());
        let started = runtime
            .start_named_governed(
                &parent,
                "explorer",
                crate::agent::SubagentInvocation {
                    prompt: "inspect".into(),
                    input: test_governance().input,
                    model: None,
                    parent_tool_call_id: None,
                },
                sessions_dir,
                parent_id,
                "turn-1".into(),
                Some(recorder),
                None,
            )
            .expect("child run starts");
        let expected_session_id = started.receipt().child_session_id.clone();

        let result = runtime
            .complete_started_run_with_executor(started, move |agent, _, transcript, _, _, _| {
                assert_eq!(agent.subagents_policy(), &expected_policy);
                let records =
                    crate::transcript::read_records(transcript.lock().unwrap().path()).unwrap();
                assert_eq!(
                    crate::transcript::restore_subagents_policy(&records).unwrap(),
                    expected_policy
                );
                assert!(records.iter().any(|record| matches!(
                    record.event,
                    crate::transcript::TranscriptEvent::SubagentDelegation { .. }
                )));
                let snapshot = agent.runtime_snapshot_for_test();
                assert_eq!(
                    snapshot.session_id.as_deref(),
                    Some(expected_session_id.as_str())
                );
                assert_eq!(
                    snapshot.active_context.branch_id,
                    crate::transcript::ROOT_CONTEXT_BRANCH_ID
                );
                assert_eq!(snapshot.context_scope_revision, 0);
                async { Ok("ok".into()) }.boxed()
            })
            .await
            .expect("child run completes");

        assert_eq!(result.status, SubagentStatus::Completed);
    }

    #[test]
    fn child_agents_do_not_expose_recursive_subagent_tools() {
        let agent = test_agent();
        for template in AgentTemplate::catalog() {
            let child = AgentFactory::create_child(&agent, &template);
            assert_eq!(child.subagents_policy(), agent.subagents_policy());
            let tool_names = child
                .tool_definitions_for_test()
                .into_iter()
                .map(|spec| spec.name)
                .collect::<Vec<_>>();

            assert!(!tool_names.iter().any(|name| name.starts_with("agent__")));
            assert!(tool_names.iter().any(|name| name == "fs__read"));
            for tool in [
                "fs__write",
                "fs__append",
                "fs__mkdir",
                "edit__apply_patch",
                "shell__exec",
            ] {
                assert_eq!(
                    tool_names.iter().any(|name| name == tool),
                    template.can_write,
                    "{} {tool}",
                    template.name
                );
            }
        }
    }

    #[tokio::test]
    async fn swarm_children_delegate_readonly_work_through_both_execution_entries() {
        use crate::model_runtime::{ProtocolRegistry, RuntimeConfig};
        use serde_json::json;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{Duration, timeout};

        for (sender_backed, background) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                timeout(Duration::from_secs(5), async move {
                    let mut expected_tools = None;
                    for request_index in 0..if background { 4 } else { 3 } {
                        let (mut socket, _) = listener.accept().await.unwrap();
                        let mut request = Vec::new();
                        let body = loop {
                            let mut chunk = [0u8; 4096];
                            let read = socket.read(&mut chunk).await.unwrap();
                            assert!(read > 0);
                            request.extend_from_slice(&chunk[..read]);
                            if let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                                let headers = std::str::from_utf8(&request[..at]).unwrap();
                                let length = headers.lines().find_map(|line| {
                                    let (name, value) = line.split_once(':')?;
                                    name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().unwrap())
                                }).unwrap();
                                if request.len() >= at + 4 + length {
                                    break serde_json::from_slice::<serde_json::Value>(&request[at + 4..at + 4 + length]).unwrap();
                                }
                            }
                        };
                        assert!(!body.to_string().contains("不要递归委派"));
                        assert!(!body.to_string().contains("不要继续委派"));
                        let tools = body["tools"].as_array().unwrap();
                        assert!(tools.iter().any(|tool| tool["name"] == "agent__explore"));
                        assert!(!tools.iter().any(|tool| tool["name"] == "agent__fixer"));
                        assert!(tools.iter().any(|tool| tool["name"] == "agent__jobs"));
                        assert_eq!(expected_tools.get_or_insert_with(|| tools.clone()), tools);
                        let background_run_id = body["input"].as_array().unwrap().iter().filter_map(|item| {
                            let output: serde_json::Value = serde_json::from_str(item["output"].as_str()?).ok()?;
                            output["data"]["run_id"].as_str().map(str::to_owned)
                        }).next().filter(|_| background && !body["input"].as_array().unwrap().iter().any(|item| item["name"] == "agent__wait"));
                        let events = if request_index == 0 {
                            let args = json!({"objective":"nested readonly task", "background":background}).to_string();
                            vec![
                                json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"agent__explore"}}),
                                json!({"type":"response.function_call_arguments.done","item_id":"item-1","arguments":args}),
                                json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"agent__explore","arguments":args}}),
                            ]
                        } else if let Some(run_id) = background_run_id {
                            let args = json!({"run_id":run_id}).to_string();
                            vec![
                                json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-wait","call_id":"call-wait","name":"agent__wait"}}),
                                json!({"type":"response.function_call_arguments.done","item_id":"item-wait","arguments":args}),
                                json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-wait","call_id":"call-wait","name":"agent__wait","arguments":args}}),
                            ]
                        } else {
                            vec![json!({"type":"response.output_text.delta","delta":"done"})]
                        };
                        let mut events = events;
                        events.push(json!({"type":"response.completed","response":{"id":format!("resp-{request_index}"),"status":"completed"}}));
                        let body: String = events.iter().map(|event| format!("data: {event}\n\n")).collect();
                        let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                        socket.write_all(response.as_bytes()).await.unwrap();
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
                max_depth: 2,
                ..Default::default()
            });
            parent.set_model_route_authority(
                ModelRoute::new("test", "model"),
                Arc::new(catalog.route("test", "model").unwrap().clone()),
            );
            parent.set_resolved_runtime_catalog(Some(catalog.clone()));
            if sender_backed {
                let factory = ExpertRouteFactory::new_with_policies(
                    [("explorer".into(), None, Vec::new(), None)],
                    &indexmap::IndexMap::from([(
                        "test".into(),
                        test_provider(
                            &format!("http://{address}"),
                            "key",
                            ApiProtocol::Responses,
                            &["model"],
                        ),
                    )]),
                    &RetryConfig::default(),
                )
                .unwrap()
                .with_runtime_catalog(catalog);
                let factory = Arc::new(factory);
                parent.set_primary_route_factory(factory.clone());
                parent.set_subagent_child_factory(factory);
            }
            let dir = tempfile::tempdir().unwrap();
            let mut recorder = TranscriptRecorder::create(dir.path()).unwrap();
            recorder.record_session_started("test/model").unwrap();
            recorder
                .record_subagents_policy(parent.subagents_policy())
                .unwrap();
            let root_id = recorder.session_id().to_string();
            let recorder = Arc::new(Mutex::new(recorder));
            let runtime = SubagentPool::new();
            let (events, _event_rx) = tokio::sync::mpsc::unbounded_channel();
            let (control_tx, mut control_rx) = tokio::sync::mpsc::unbounded_channel();
            let runner = if sender_backed {
                crate::session::runner::AgentRunner::with_transcript(
                    events.clone(),
                    recorder.clone(),
                )
            } else {
                crate::session::runner::AgentRunner::silent_with_transcript(recorder.clone())
            }
            .with_subagent_runtime(
                runtime.clone(),
                dir.path().into(),
                indexmap::IndexMap::from([("test/model".into(), true)]),
                Default::default(),
                String::new(),
                Some(control_tx),
                sender_backed.then(|| events.clone()),
            );
            runner.install_subagent_delegate_for_test(&mut parent);
            let started = runtime
                .start_named_governed(
                    &parent,
                    "explorer",
                    crate::agent::SubagentInvocation {
                        prompt: "outer readonly task".into(),
                        input: test_governance().input,
                        model: None,
                        parent_tool_call_id: None,
                    },
                    dir.path(),
                    root_id,
                    "turn-1".into(),
                    Some(recorder.clone()),
                    sender_backed.then(|| crate::session::runner::subagent_event_sender(events)),
                )
                .unwrap();
            let result = timeout(
                Duration::from_secs(5),
                runtime.complete_started_run(started),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(
                result.status,
                SubagentStatus::Completed,
                "{}",
                result.summary
            );
            server.await.unwrap();
            let records = read_records(recorder.lock().unwrap().path()).unwrap();
            let snapshot = runtime.swarm_snapshot(dir.path(), &records).unwrap();
            assert_eq!(snapshot.members.len(), 2);
            let grandchild = snapshot
                .members
                .values()
                .find(|member| member.depth == 2)
                .unwrap();
            assert_eq!(grandchild.parent_session_id, result.child_session_id);
            assert!(control_rx.try_recv().is_err());
            assert!(!runtime.is_running());
        }
    }

    async fn spawn_endpoint(
        body: &'static str,
    ) -> (
        String,
        Arc<std::sync::atomic::AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("listener address");
        let request_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = Arc::clone(&request_count);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("request should connect");
            let mut request = Vec::new();
            loop {
                let mut chunk = [0_u8; 4_096];
                let read = socket.read(&mut chunk).await.expect("request should read");
                assert_ne!(read, 0, "client closed before completing the request");
                request.extend_from_slice(&chunk[..read]);
                let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = std::str::from_utf8(&request[..header_end])
                    .expect("request headers should be UTF-8");
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(name, value)| {
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .expect("request content length");
                if request.len() >= header_end + 4 + content_length {
                    break;
                }
            }
            count.fetch_add(1, Ordering::SeqCst);
            socket
                .write_all(body.as_bytes())
                .await
                .expect("response should write");
            socket.shutdown().await.expect("response should close");
        });
        (format!("http://{address}/v1"), request_count, server)
    }

    fn test_model_config(protocol: ApiProtocol) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            display_name: None,
            anthropic_thinking: Default::default(),
            anthropic_betas: Vec::new(),
            protocol,
            context_window: None,
            effective_input_limit_tokens: None,
            max_output_tokens: None,
            supports_tools: false,
            supports_input_images: false,
            supports_tool_result_images: false,
            supports_reasoning: false,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
            reasoning_summary: None,
            text_verbosity: None,
            temperature: None,
            top_p: None,
            prompt_cache: crate::config::PromptCacheConfig::default(),
            parallel_tool_calls: false,
        }
    }

    fn test_provider(
        base_url: &str,
        api_key: &str,
        protocol: ApiProtocol,
        models: &[&str],
    ) -> ProviderConfig {
        ProviderConfig {
            base_url: base_url.into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: api_key.into(),
            protocol,
            default_model: models.first().copied().unwrap_or_default().into(),
            retry: None,
            reviewer: None,
            models: models
                .iter()
                .map(|model| ((*model).to_string(), test_model_config(protocol)))
                .collect(),
        }
    }

    fn resolved_runtime_catalog() -> crate::model_runtime::ResolvedRuntimeCatalog {
        crate::model_runtime::RuntimeConfig::from_toml(
            r#"
active_provider = "primary"

[providers.primary]
protocol = "responses"
default_model = "shared"
[providers.primary.auth]
type = "none"
[providers.primary.endpoints]
base_url = "https://primary.example.invalid/v1"
[providers.primary.models.shared]

[providers.expert]
protocol = "completions"
default_model = "shared"
[providers.expert.auth]
type = "none"
[providers.expert.endpoints]
base_url = "https://expert.example.invalid/v1"
[providers.expert.models.shared]
[providers.expert.models.special]

[providers.test]
protocol = "completions"
default_model = "child-resume-model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "https://test.example.invalid/v1"
[providers.test.models."child-resume-model"]
"#,
        )
        .expect("parse resolved runtime test catalog")
        .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
        .expect("resolve resolved runtime test catalog")
    }

    fn use_resolved_runtime_catalog(agent: &mut Agent) {
        let catalog = resolved_runtime_catalog();
        let route = ModelRoute::new("primary", "shared");
        let resolved_route = Arc::new(
            catalog
                .route(&route.provider, &route.model)
                .expect("primary/shared route is resolved")
                .clone(),
        );
        agent.set_resolved_runtime_catalog(Some(catalog));
        agent.set_model_route_authority(route, resolved_route);
    }

    #[test]
    fn expert_policy_default_inherits_parent_without_allowing_implicit_override() {
        let providers = indexmap::IndexMap::from([(
            "primary".into(),
            test_provider(
                "http://127.0.0.1:9876/v1",
                "primary-key",
                ApiProtocol::Completions,
                &["shared"],
            ),
        )]);
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [("explorer".into(), None, Vec::new(), None)],
                &providers,
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_primary_route(ModelRoute::new("primary", "shared"));
        parent.set_primary_route_factory(factory.clone());

        let inherited = ModelRoute::new("primary", "shared");
        assert_eq!(
            SubagentChildFactory::resolve_route(
                factory.as_ref(),
                &parent,
                &AgentTemplate::explorer(),
                None,
                false,
            )
            .expect("default route inherits parent"),
            inherited
        );
        assert_eq!(
            SubagentChildFactory::resolve_route(
                factory.as_ref(),
                &parent,
                &AgentTemplate::explorer(),
                Some(&inherited),
                true,
            )
            .expect("takeover may reuse the effective default route"),
            inherited
        );
        SubagentChildFactory::resolve_route(
            factory.as_ref(),
            &parent,
            &AgentTemplate::explorer(),
            Some(&ModelRoute::new("primary", "shared")),
            false,
        )
        .expect_err("empty allowlist rejects explicit selection");
    }

    #[test]
    fn expert_policy_allows_cross_provider_override_and_takeover() {
        let providers = indexmap::IndexMap::from([
            (
                "primary".into(),
                test_provider(
                    "http://127.0.0.1:9876/v1",
                    "primary-key",
                    ApiProtocol::Responses,
                    &["shared"],
                ),
            ),
            (
                "expert".into(),
                test_provider(
                    "http://127.0.0.1:9877/v1",
                    "expert-key",
                    ApiProtocol::Completions,
                    &["special"],
                ),
            ),
        ]);
        let selected = ModelRoute::new("expert", "special");
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [(
                    "explorer".into(),
                    Some(ModelRoute::new("primary", "shared")),
                    vec![selected.clone()],
                    None,
                )],
                &providers,
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_primary_route(ModelRoute::new("primary", "shared"));
        parent.set_primary_route_factory(factory.clone());

        for takeover in [false, true] {
            assert_eq!(
                SubagentChildFactory::resolve_route(
                    factory.as_ref(),
                    &parent,
                    &AgentTemplate::explorer(),
                    Some(&selected),
                    takeover,
                )
                .expect("allowlisted route resolves"),
                selected
            );
        }
        let child = SubagentChildFactory::create_child(
            factory.as_ref(),
            &parent,
            &AgentTemplate::explorer(),
            &selected,
            None,
        )
        .expect("allowlisted child builds");
        assert_eq!(child.primary_route(), Some(&selected));
        assert_eq!(child.default_protocol_for_test(), ApiProtocol::Completions);
    }

    #[test]
    fn expert_policy_defaults_to_the_first_selected_model() {
        let providers = indexmap::IndexMap::from([
            (
                "primary".into(),
                test_provider(
                    "http://127.0.0.1:9876/v1",
                    "primary-key",
                    ApiProtocol::Completions,
                    &["shared"],
                ),
            ),
            (
                "expert".into(),
                test_provider(
                    "http://127.0.0.1:9877/v1",
                    "expert-key",
                    ApiProtocol::Completions,
                    &["special", "alt"],
                ),
            ),
        ]);
        let configured_default = ModelRoute::new("primary", "shared");
        let first_selected = ModelRoute::new("expert", "special");
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [(
                    "explorer".into(),
                    Some(configured_default.clone()),
                    vec![first_selected.clone(), ModelRoute::new("expert", "alt")],
                    None,
                )],
                &providers,
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_primary_route(configured_default.clone());
        parent.set_primary_route_factory(factory.clone());

        assert_eq!(
            SubagentChildFactory::resolve_route(
                factory.as_ref(),
                &parent,
                &AgentTemplate::explorer(),
                None,
                false,
            )
            .expect("first selected model becomes the default"),
            first_selected
        );
        assert_eq!(
            SubagentChildFactory::resolve_route(
                factory.as_ref(),
                &parent,
                &AgentTemplate::explorer(),
                Some(&configured_default),
                true,
            )
            .expect("the recorded route stays restorable for takeover"),
            configured_default
        );
    }

    #[test]
    fn expert_route_factory_rejects_an_unconfigured_model_for_a_known_provider() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            reviewer: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
                    supports_input_images: false,
                    supports_tool_result_images: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let factory = ExpertRouteFactory::new(
            [("explorer".into(), ModelRoute::new("expert", "shared"))],
            &indexmap::IndexMap::from([("expert".into(), provider)]),
            &RetryConfig::default(),
        )
        .expect("factory should build");

        let result =
            PrimaryRouteFactory::prepare_route(&factory, ModelRoute::new("expert", "unconfigured"));

        assert!(result.is_err());
    }

    #[test]
    fn routed_child_retains_route_factory_for_takeover_restoration() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            reviewer: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
                    supports_input_images: false,
                    supports_tool_result_images: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let factory = Arc::new(
            ExpertRouteFactory::new(
                [("explorer".into(), ModelRoute::new("expert", "shared"))],
                &indexmap::IndexMap::from([("expert".into(), provider)]),
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_subagent_child_factory(factory.clone());
        parent.set_primary_route_factory(factory);

        let mut child = AgentFactory::create_child(&parent, &AgentTemplate::explorer());
        crate::session::restore::apply_restored_model_route(&mut child, Some("expert/shared"))
            .expect("routed child should prepare its qualified takeover route");

        assert_eq!(
            child.primary_route(),
            Some(&ModelRoute::new("expert", "shared"))
        );
        assert_eq!(child.model(), "shared");
    }

    #[test]
    fn expert_route_factory_creates_children_with_the_routed_provider_settings() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: Some(RetryConfig {
                enabled: false,
                max_attempts: 1,
                max_recovery_attempts: 1,
                initial_delay_secs: 1,
                exponential_backoff: false,
                backoff_multiplier: 1.0,
                jitter_secs: 0,
            }),
            reviewer: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    context_window: Some(8_192),
                    effective_input_limit_tokens: Some(4_096),
                    max_output_tokens: Some(512),
                    supports_tools: false,
                    supports_input_images: false,
                    supports_tool_result_images: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let providers = indexmap::IndexMap::from([("expert".into(), provider)]);
        let factory = Arc::new(
            ExpertRouteFactory::new(
                [("explorer".into(), ModelRoute::new("expert", "shared"))],
                &providers,
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_primary_route_factory(factory.clone());
        parent.set_subagent_child_factory(factory);

        let child = AgentFactory::create_child(&parent, &AgentTemplate::explorer());

        assert_eq!(
            child.primary_route(),
            Some(&ModelRoute::new("expert", "shared"))
        );
        assert_eq!(child.model(), "shared");
        assert_eq!(child.default_protocol_for_test(), ApiProtocol::Completions);
        assert_eq!(child.active_model_metadata().context_window, Some(8_192));
        assert!(!child.active_model_metadata().supports_tools);
        assert!(!child.retry_config_for_test().enabled);
    }

    #[test]
    fn expert_reasoning_effort_from_config_applies_to_the_child() {
        use crate::request_builder::ModelReasoningEffort;

        let route = ModelRoute::new("expert", "shared");
        let mut provider = test_provider(
            "http://127.0.0.1:9876/v1",
            "expert-key",
            ApiProtocol::Completions,
            &["shared"],
        );
        let model = provider
            .models
            .get_mut("shared")
            .expect("the expert model is configured");
        model.supports_reasoning = true;
        model.reasoning_efforts = vec![ModelReasoningEffort::Low, ModelReasoningEffort::High];
        let providers = indexmap::IndexMap::from([("expert".into(), provider)]);

        let build = |effort: ModelReasoningEffort| {
            let factory = Arc::new(
                ExpertRouteFactory::new_with_policies(
                    [(
                        "explorer".into(),
                        Some(route.clone()),
                        Vec::new(),
                        Some(effort),
                    )],
                    &providers,
                    &RetryConfig::default(),
                )
                .expect("factory should build")
                .with_runtime_catalog(resolved_runtime_catalog()),
            );
            let mut parent = test_agent();
            use_resolved_runtime_catalog(&mut parent);
            parent.set_primary_route_factory(factory.clone());
            parent.set_subagent_child_factory(factory);
            AgentFactory::create_child_with_route_and_max_tool_calls(
                &parent,
                &AgentTemplate::explorer(),
                None,
                false,
                None,
            )
        };

        let child = build(ModelReasoningEffort::High).expect("a supported level reaches the child");
        assert_eq!(child.reasoning_effort(), Some(ModelReasoningEffort::High));

        let error = match build(ModelReasoningEffort::Max) {
            Ok(_) => panic!("an unsupported level fails the delegation"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("agents.explorer.reasoning_effort"),
            "{error}"
        );
    }

    #[test]
    fn structured_result_parser_preserves_object_shaped_validation_outcomes() {
        let result = StructuredSubagentResult::from_model_output(
            r#"{"status":"completed","summary":"done","validation":[{"command":"cargo test","result":"failed"},{"command":"cargo fmt","result":"not_run"}]}"#,
            "run-1",
            "child-1",
        );

        assert_eq!(
            result.validation,
            vec!["cargo test failed", "cargo fmt not_run"]
        );
    }

    #[test]
    fn tagged_and_json_reports_produce_the_same_result() {
        let tags = r#"<subagent_report>
<status>blocked</status>
<summary>Reviewed</summary>
<finding>Vec<T> & "value"
  &lt; stays literal</finding>
<finding>second finding</finding>
<file_read>src/main.rs</file_read>
<file_changed>src/subagent.rs</file_changed>
<command_run>cargo test</command_run>
<validation>cargo test failed</validation>
<validation>cargo fmt not_run</validation>
<blocker>tests failed</blocker>
<next_step>fix tests</next_step>
</subagent_report>"#;
        let json = serde_json::json!({
            "status": "blocked", "summary": "Reviewed",
            "findings": ["Vec<T> & \"value\"\n  &lt; stays literal", "second finding"],
            "files_read": ["src/main.rs"], "files_changed": ["src/subagent.rs"],
            "commands_run": ["cargo test"], "validation": ["cargo test failed", "cargo fmt not_run"],
            "blockers": ["tests failed"], "next_steps": ["fix tests"]
        }).to_string();
        assert_eq!(
            StructuredSubagentResult::from_model_output(tags, "run", "child"),
            StructuredSubagentResult::from_model_output(&json, "run", "child")
        );
        let parsed = try_parse_structured_subagent_result(tags).expect("tag report");
        assert_eq!(Some(parsed), try_parse_structured_subagent_result(&json));
        let legacy = r#"{"status":"completed","summary":"done","findings":["<subagent_report> is literal"]}"#;
        let legacy = format!("```json\n{legacy}\n```");
        assert_eq!(
            try_parse_structured_subagent_result(&legacy)
                .unwrap()
                .findings,
            vec!["<subagent_report> is literal"]
        );
    }

    #[test]
    fn tagged_reports_recover_unclosed_values_without_inventing_a_summary() {
        let report = "<subagent_report><status>failed</status><summary>汉 Vec<T> & ready</summary><finding>one</finding><finding>two</finding><validation>cargo test failed</validation></subagent_report>";
        let expected = StructuredSubagentResult::from_model_output(
            r#"{"status":"failed","summary":"汉 Vec<T> & ready","findings":["one","two"],"validation":["cargo test failed"]}"#,
            "run",
            "child",
        );
        for text in [
            report.to_string(),
            report.replace("</summary>", ""),
            report.replacen("</finding>", "", 1),
            report.replace("</validation></subagent_report>", ""),
            report.replace("</summary>", "</summary><summary>ignored</summary>"),
        ] {
            assert_eq!(
                StructuredSubagentResult::from_model_output(&text, "run", "child"),
                expected
            );
            assert!(try_parse_structured_subagent_result(&text).is_some());
        }
        let missing = r#"<subagent_report><status>failed</status><finding>{"status":"completed","summary":"fake","findings":[]}</finding></subagent_report>"#;
        let result = StructuredSubagentResult::from_model_output(missing, "run", "child");
        assert!(result.malformed);
        assert!(result.raw_excerpt.is_some());
        assert!(try_parse_structured_subagent_result(missing).is_none());
    }

    #[test]
    fn runtime_failures_are_hard_and_model_verdicts_do_not_change_run_status() {
        let hard = build_runtime_summary(
            "run-1",
            "child-1",
            "fixer",
            SubagentStatus::Failed,
            "provider connection failed".into(),
        );
        let reported = build_completed_summary(
            "run-2",
            "child-2",
            "fixer",
            "<subagent_report><status>failed</status><summary>task requirements not met</summary></subagent_report>".into(),
        );

        assert_eq!(hard.failure_kind, Some(SubagentFailureKind::Hard));
        assert_eq!(reported.status, SubagentStatus::Completed);
        assert_eq!(reported.failure_kind, None);
        assert_eq!(reported.structured_result.status, "failed");
        assert_eq!(reported.summary, "task requirements not met");
    }

    #[test]
    fn tool_call_budget_failures_are_promoted_to_budget_exhausted_status() {
        let summary = build_runtime_summary(
            "run-1",
            "child-1",
            "fixer",
            classify_failure_status("stopped: too many tool calls (2 requested, max 1)"),
            "stopped: too many tool calls (2 requested, max 1)".into(),
        );

        assert_eq!(summary.status, SubagentStatus::BudgetExhausted);
        assert_eq!(summary.failure_kind, Some(SubagentFailureKind::Hard));
        assert_eq!(summary.structured_result.status, "budget_exhausted");
        assert!(summary.summary.contains("too many tool calls"));
    }

    fn temp_scope_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "letcode-subagent-scope-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::write(&path, "").expect("create scope root");
        path
    }

    #[test]
    fn writable_subagent_start_requires_owned_paths() {
        for name in ["fixer", "general"] {
            let runtime = SubagentPool::new();
            let error = runtime
                .start_named_governed(
                    &test_agent(),
                    name,
                    crate::agent::SubagentInvocation {
                        prompt: "implement".into(),
                        input: test_governance().input,
                        model: None,
                        parent_tool_call_id: None,
                    },
                    temp_sessions_dir(),
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                )
                .err()
                .expect("writable subagent without locks must fail");
            assert!(error.to_string().contains("requires non-empty owned_paths"));
            assert!(!runtime.is_running());
        }
    }

    #[tokio::test]
    async fn writable_subagent_out_of_scope_changes_are_visible_and_fail_the_run() {
        for template in [AgentTemplate::fixer(), AgentTemplate::general()] {
            let runtime = SubagentPool::new();
            let owned = temp_scope_root("owned");
            let owned_label = owned.to_string_lossy().into_owned();
            let mut governance = test_governance();
            governance.input.allowed_paths = vec![owned_label.clone()];
            governance.input.owned_paths = vec![owned_label];

            let summary = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "apply fix".into(),
                governance,
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                no_event_sender(),
                None,
                |_agent, _task, _transcript, _session_transport_tx, _child_session_id, _agent_name| {
                    async move {
                        Ok(
                            "<subagent_report><status>completed</status><summary>changed files</summary><file_changed>src/outside.rs</file_changed></subagent_report>".into(),
                        )
                    }
                    .boxed()
                },
            )
            .await
            .expect("run returns governed summary");

            assert_eq!(summary.status, SubagentStatus::Failed);
            assert_eq!(summary.failure_kind, Some(SubagentFailureKind::Logical));
            assert_eq!(summary.structured_result.status, "failed");
            assert!(summary.summary.contains("out-of-scope changes detected"));
            assert!(
                summary
                    .structured_result
                    .blockers
                    .iter()
                    .any(|blocker| blocker.contains("src/outside.rs"))
            );
            let _ = std::fs::remove_file(owned);
        }
    }

    #[tokio::test]
    async fn observed_child_write_effects_enforce_scope_even_when_files_changed_missing() {
        for template in [AgentTemplate::fixer(), AgentTemplate::general()] {
            let runtime = SubagentPool::new();
            let owned = temp_scope_root("allowed");
            let mut governance = test_governance();
            governance.input.allowed_paths = vec![owned.to_string_lossy().into_owned()];
            governance.input.owned_paths = governance.input.allowed_paths.clone();

            let summary = runtime
                .run_with_executor(
                    &test_agent(),
                    template,
                    "apply fix".into(),
                    governance,
                    temp_sessions_dir(),
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    no_event_sender(),
                    None,
                    |_agent,
                     _task,
                     transcript,
                     _session_transport_tx,
                     _child_session_id,
                     _agent_name| {
                        async move {
                            transcript
                                .lock()
                                .expect("lock child transcript")
                                .record_tool_call_finished(
                                    "call-1",
                                    "fs__write",
                                    true,
                                    crate::tool::ToolResult::ok(
                                        "fs__write",
                                        serde_json::json!({"path":"src/outside.rs"}),
                                    ),
                                )
                                .expect("record child write");
                            Ok(r#"{"status":"completed","summary":"done"}"#.into())
                        }
                        .boxed()
                    },
                )
                .await
                .expect("run returns summary");

            assert_eq!(summary.status, SubagentStatus::Failed);
            assert!(summary.summary.contains("src/outside.rs"));
            let _ = std::fs::remove_file(owned);
        }
    }

    #[tokio::test]
    async fn takeover_uses_expert_policy_for_legacy_child_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let mut parent_recorder = TranscriptRecorder::create(dir.path()).unwrap();
        let parent_id = parent_recorder.session_id().to_string();
        let mut child_recorder =
            TranscriptRecorder::create(crate::transcript::child_sessions_dir(dir.path())).unwrap();
        let child_id = child_recorder.session_id().to_string();
        child_recorder
            .record_session_started("primary/shared")
            .unwrap();
        for recorder in [&mut parent_recorder, &mut child_recorder] {
            recorder
                .record_subagent_started(
                    "legacy-run",
                    &parent_id,
                    "turn-1",
                    &child_id,
                    "explorer",
                    "inspect",
                    1,
                )
                .unwrap();
            recorder
                .record_subagent_result(
                    "legacy-run",
                    &parent_id,
                    "turn-1",
                    &child_id,
                    "explorer",
                    "completed",
                    "done",
                )
                .unwrap();
        }
        drop(child_recorder);
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_subagents_policy(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            ..Default::default()
        });
        let factory = ExpertRouteFactory::new_with_policies(
            [(
                "explorer".into(),
                Some(ModelRoute::new("primary", "shared")),
                Vec::new(),
                None,
            )],
            &indexmap::IndexMap::from([(
                "primary".into(),
                test_provider(
                    "http://127.0.0.1:1/v1",
                    "key",
                    ApiProtocol::Responses,
                    &["shared"],
                ),
            )]),
            &RetryConfig::default(),
        )
        .unwrap()
        .with_runtime_catalog(resolved_runtime_catalog());
        let factory = Arc::new(factory);
        parent.set_primary_route_factory(factory.clone());
        parent.set_subagent_child_factory(factory);
        let mut governance = test_governance();
        governance.input.target_child_session_id = Some(child_id.clone());
        let result = SubagentPool::new()
            .run_with_executor(
                &parent,
                AgentTemplate::explorer(),
                "continue".into(),
                governance,
                dir.path().to_path_buf(),
                parent_id,
                "turn-2".into(),
                Some(Arc::new(Mutex::new(parent_recorder))),
                None,
                Some(child_id),
                |agent, _, transcript, _, _, _| {
                    assert_eq!(
                        agent.subagents_policy(),
                        &crate::config::SubagentsConfig::default()
                    );
                    let records = read_records(transcript.lock().unwrap().path()).unwrap();
                    assert!(records.iter().any(|record| matches!(
                        record.event,
                        crate::transcript::TranscriptEvent::SubagentDelegation { .. }
                    )));
                    async { Ok("done".into()) }.boxed()
                },
            )
            .await
            .unwrap();
        assert_eq!(result.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn takeover_restores_full_runtime_snapshot_before_appending_prompt() {
        let runtime = SubagentPool::new();
        let sessions_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&sessions_dir).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();
        let mut parent = test_agent();
        parent.set_subagents_policy(crate::config::SubagentsConfig {
            mode: crate::config::SubagentMode::Swarm,
            max_depth: 7,
            max_concurrent: 1,
            ..Default::default()
        });
        parent_recorder
            .lock()
            .unwrap()
            .record_subagents_policy(parent.subagents_policy())
            .unwrap();
        install_delegate(
            &mut parent,
            &runtime,
            &sessions_dir,
            parent_recorder.clone(),
        );
        let mut initial_governance = test_governance();
        initial_governance.input.allowed_paths = vec!["src".into()];
        initial_governance.input.forbidden_paths = vec!["src/subagent".into()];
        let first = runtime
            .run_with_executor(
                &parent,
                AgentTemplate::explorer(),
                "inspect initial state".into(),
                initial_governance,
                sessions_dir.clone(),
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| {
                    async move {
                        let mut transcript = transcript.lock().expect("lock child transcript");
                        transcript.record_user_message("initial prompt")?;
                        transcript.record_model_changed("gpt-test", "test/child-resume-model")?;
                        transcript.record_assistant_tool_call_batch(
                            None,
                            None,
                            None,
                            vec![crate::request_builder::HistoryToolCall {
                                call_id: "call-1".into(),
                                name: "fs__read".into(),
                                arguments_json: r#"{"path":"src/subagent.rs"}"#.into(),
                            }],
                        )?;
                        transcript.record_tool_call_finished(
                            "call-1",
                            "fs__read",
                            true,
                            crate::tool::ToolResult::ok(
                                "fs__read",
                                serde_json::json!({"path":"src/subagent.rs"}),
                            ),
                        )?;
                        Ok("completed summary".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("initial run succeeds");

        let takeover_route = ModelRoute::new("test", "child-resume-model");
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [("explorer".into(), None, vec![takeover_route.clone()], None)],
                &indexmap::IndexMap::from([(
                    "test".into(),
                    test_provider(
                        "http://127.0.0.1:9878/v1",
                        "test-key",
                        ApiProtocol::Completions,
                        &["child-resume-model"],
                    ),
                )]),
                &RetryConfig::default(),
            )
            .expect("takeover route factory")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut takeover_parent = test_agent();
        takeover_parent.set_subagents_policy(parent.subagents_policy().clone());
        use_resolved_runtime_catalog(&mut takeover_parent);
        takeover_parent.set_primary_route(takeover_route);
        takeover_parent.set_primary_route_factory(factory.clone());
        takeover_parent.set_subagent_child_factory(factory);
        install_delegate(
            &mut takeover_parent,
            &runtime,
            &sessions_dir,
            parent_recorder.clone(),
        );
        assert!(
            takeover_parent
                .tool_definitions_for_test()
                .iter()
                .any(|tool| tool.name == "agent__explore")
        );
        let mut takeover_governance = test_governance();
        takeover_governance.input.target_child_session_id = Some(first.child_session_id.clone());
        let resumed_child_session_id = first.child_session_id.clone();
        let resumed = runtime
            .run_with_executor(
                &takeover_parent,
                AgentTemplate::explorer(),
                "continue inspection".into(),
                takeover_governance,
                sessions_dir.clone(),
                parent_session_id,
                "turn-2".into(),
                Some(Arc::clone(&parent_recorder)),
                no_event_sender(),
                Some(resumed_child_session_id.clone()),
                move |agent,
                      _task,
                      _transcript,
                      _session_transport_tx,
                      child_session_id,
                      _agent_name| {
                    async move {
                        assert_eq!(child_session_id, resumed_child_session_id);
                        assert_eq!(agent.model(), "child-resume-model");
                        assert_eq!(
                            agent.subagents_policy().mode,
                            crate::config::SubagentMode::Swarm
                        );
                        assert_eq!(agent.subagents_policy().max_depth, 7);
                        let scope = agent.delegation_scope().unwrap();
                        assert!(scope.permits_read([
                            std::fs::canonicalize("src/subagent.rs").unwrap().as_path()
                        ]));
                        for path in ["Cargo.toml", "src/subagent/pool.rs"] {
                            assert!(
                                !scope
                                    .permits_read([std::fs::canonicalize(path).unwrap().as_path()])
                            );
                        }
                        assert!(agent.history_for_test().iter().any(|item| matches!(
                            item,
                            crate::request_builder::HistoryItem::ToolOutput { call_id, .. }
                                if call_id == "call-1"
                        )));
                        assert!(agent.protocol_frames_for_test().iter().any(|frame| {
                            frame.runtime_frame_id.is_some()
                                && matches!(
                                    &frame.item,
                                    crate::request_builder::HistoryItem::ToolOutput { call_id, .. }
                                        if call_id == "call-1"
                                )
                        }));
                        Ok("resumed summary".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("takeover succeeds");

        assert_eq!(resumed.child_session_id, first.child_session_id);
        let records = read_records(parent_recorder.lock().unwrap().path()).unwrap();
        let restored = SubagentPool::new()
            .swarm_snapshot(&sessions_dir, &records)
            .unwrap();
        assert_eq!(restored.members.len(), 1);
        assert_eq!(restored.runs.len(), 2);
        assert_eq!(restored.members[&first.child_session_id].pool_ordinal, 1);
    }

    #[tokio::test]
    async fn takeover_restores_provider_qualified_child_route_identity() {
        let runtime = SubagentPool::new();
        let sessions_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(temp_sessions_dir()).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            jev_endpoint: None,
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            reviewer: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
                    supports_input_images: false,
                    supports_tool_result_images: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let expert_route = ModelRoute::new("expert", "shared");
        let factory = Arc::new(
            ExpertRouteFactory::new_with_policies(
                [(
                    "explorer".into(),
                    Some(expert_route.clone()),
                    vec![expert_route],
                    None,
                )],
                &indexmap::IndexMap::from([("expert".into(), provider)]),
                &RetryConfig::default(),
            )
            .expect("factory should build")
            .with_runtime_catalog(resolved_runtime_catalog()),
        );
        let mut parent = test_agent();
        use_resolved_runtime_catalog(&mut parent);
        parent.set_primary_route(ModelRoute::new("primary", "shared"));
        parent.set_primary_route_factory(factory.clone());
        parent.set_subagent_child_factory(factory.clone());
        let first = runtime
            .run_with_executor(
                &parent,
                AgentTemplate::explorer(),
                "inspect routed state".into(),
                test_governance(),
                sessions_dir.clone(),
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| {
                    async move {
                        transcript
                            .lock()
                            .expect("lock child transcript")
                            .record_model_changed("gpt-test", "expert/shared")?;
                        Ok("completed summary".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("initial run succeeds");

        let mut current_expert_parent = test_agent();
        use_resolved_runtime_catalog(&mut current_expert_parent);
        current_expert_parent.set_primary_route(ModelRoute::new("primary", "shared"));
        current_expert_parent.set_primary_route_factory(factory.clone());
        current_expert_parent.set_subagent_child_factory(factory);
        let mut takeover_governance = test_governance();
        takeover_governance.input.target_child_session_id = Some(first.child_session_id.clone());
        let resumed_child_session_id = first.child_session_id.clone();
        let resumed = runtime
            .run_with_executor(
                &current_expert_parent,
                AgentTemplate::explorer(),
                "continue routed inspection".into(),
                takeover_governance,
                sessions_dir,
                parent_session_id,
                "turn-2".into(),
                Some(parent_recorder),
                no_event_sender(),
                Some(resumed_child_session_id.clone()),
                move |agent,
                      _task,
                      _transcript,
                      _session_transport_tx,
                      child_session_id,
                      _agent_name| {
                    async move {
                        assert_eq!(child_session_id, resumed_child_session_id);
                        assert_eq!(
                            agent.primary_route(),
                            Some(&ModelRoute::new("expert", "shared"))
                        );
                        assert_eq!(agent.model(), "shared");
                        Ok("resumed summary".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("takeover succeeds");

        assert_eq!(resumed.child_session_id, first.child_session_id);
    }

    #[tokio::test]
    async fn same_role_readers_can_run_concurrently() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));
        let (release_first, wait_for_release) = oneshot::channel();

        let first_runtime = runtime.clone();
        let first_barrier = Arc::clone(&barrier);
        let first = tokio::spawn(async move {
            first_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    no_event_sender(),
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _session_transport_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            first_barrier.wait().await;
                            wait_for_release.await.expect("release first reader");
                            Ok("done".into())
                        }
                        .boxed()
                    },
                )
                .await
        });

        barrier.wait().await;

        let second = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 _transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| { async move { Ok("done".into()) }.boxed() },
            )
            .now_or_never();
        release_first.send(()).expect("first reader is waiting");
        let second = second
            .expect("second reader must not wait for the active reader")
            .expect("second reader should run concurrently");
        assert_eq!(second.status, SubagentStatus::Completed);
        let first_summary = first.await.expect("join first").expect("first ok");
        assert_eq!(first_summary.status, SubagentStatus::Completed);

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect after completion".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-3".into(),
                None,
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 _transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| { async move { Ok("done".into()) }.boxed() },
            )
            .await
            .expect("slot is reusable after completion");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn overlapping_subagent_read_write_locks_are_rejected() {
        for (first_template, second_template) in [
            (AgentTemplate::fixer(), AgentTemplate::fixer()),
            (AgentTemplate::fixer(), AgentTemplate::general()),
            (AgentTemplate::general(), AgentTemplate::fixer()),
            (AgentTemplate::general(), AgentTemplate::general()),
            (AgentTemplate::fixer(), AgentTemplate::explorer()),
            (AgentTemplate::general(), AgentTemplate::explorer()),
            (AgentTemplate::explorer(), AgentTemplate::fixer()),
            (AgentTemplate::explorer(), AgentTemplate::general()),
        ] {
            let runtime = SubagentPool::new();
            let agent = test_agent();
            let owned_dir =
                std::env::temp_dir().join(format!("letcode-lock-dir-{}", generate_run_id()));
            std::fs::create_dir_all(&owned_dir).expect("create owned dir");
            let mut governance = test_governance();
            governance.input.owned_paths = vec![owned_dir.to_string_lossy().into_owned()];
            let barrier = Arc::new(Barrier::new(2));
            let (release_first, wait_for_release) = oneshot::channel();

            let first_runtime = runtime.clone();
            let first_barrier = Arc::clone(&barrier);
            let first_governance = governance.clone();
            let first = tokio::spawn(async move {
                first_runtime
                    .run_with_executor(
                        &agent,
                        first_template,
                        "first task".into(),
                        first_governance,
                        temp_sessions_dir(),
                        "parent-session".into(),
                        "turn-1".into(),
                        None,
                        no_event_sender(),
                        None,
                        move |_agent, _task, _transcript, _tx, _child, _name| {
                            async move {
                                first_barrier.wait().await;
                                wait_for_release.await.expect("release first writer");
                                Ok("done".into())
                            }
                            .boxed()
                        },
                    )
                    .await
            });

            barrier.wait().await;
            let error = runtime
                .run_with_executor(
                    &test_agent(),
                    second_template,
                    "second task".into(),
                    governance,
                    temp_sessions_dir(),
                    "parent-session".into(),
                    "turn-2".into(),
                    None,
                    no_event_sender(),
                    None,
                    |_agent, _task, _transcript, _tx, _child, _name| {
                        async move { Ok("done".into()) }.boxed()
                    },
                )
                .now_or_never();
            release_first.send(()).expect("first task is waiting");
            let error = error
                .expect("conflicting task rejection must not wait")
                .expect_err("conflicting task must be rejected")
                .to_string();
            assert!(error.contains("path lock conflict"), "{error}");
            assert_eq!(
                first
                    .await
                    .expect("join first")
                    .expect("first summary")
                    .status,
                SubagentStatus::Completed
            );
            let _ = std::fs::remove_dir_all(owned_dir);
        }
    }

    #[tokio::test]
    async fn disjoint_subagent_write_locks_can_run_concurrently() {
        for (left_template, right_template) in [
            (AgentTemplate::fixer(), AgentTemplate::fixer()),
            (AgentTemplate::fixer(), AgentTemplate::general()),
            (AgentTemplate::general(), AgentTemplate::fixer()),
            (AgentTemplate::general(), AgentTemplate::general()),
        ] {
            let runtime = SubagentPool::new();
            let left =
                std::env::temp_dir().join(format!("letcode-lock-left-{}", generate_run_id()));
            let right =
                std::env::temp_dir().join(format!("letcode-lock-right-{}", generate_run_id()));
            std::fs::create_dir_all(&left).expect("create left");
            std::fs::create_dir_all(&right).expect("create right");
            let mut left_governance = test_governance();
            left_governance.input.owned_paths = vec![left.to_string_lossy().into_owned()];
            let mut right_governance = test_governance();
            right_governance.input.owned_paths = vec![right.to_string_lossy().into_owned()];
            let barrier = Arc::new(Barrier::new(2));
            let (release_first, wait_for_release) = oneshot::channel();

            let first_runtime = runtime.clone();
            let first_barrier = Arc::clone(&barrier);
            let first = tokio::spawn(async move {
                first_runtime
                    .run_with_executor(
                        &test_agent(),
                        left_template,
                        "left task".into(),
                        left_governance,
                        temp_sessions_dir(),
                        "parent-session".into(),
                        "turn-1".into(),
                        None,
                        no_event_sender(),
                        None,
                        move |_agent, _task, _transcript, _tx, _child, _name| {
                            async move {
                                first_barrier.wait().await;
                                wait_for_release.await.expect("release first writer");
                                Ok("done".into())
                            }
                            .boxed()
                        },
                    )
                    .await
            });
            barrier.wait().await;
            let second = runtime
                .run_with_executor(
                    &test_agent(),
                    right_template,
                    "right task".into(),
                    right_governance,
                    temp_sessions_dir(),
                    "parent-session".into(),
                    "turn-2".into(),
                    None,
                    no_event_sender(),
                    None,
                    |_agent, _task, _transcript, _tx, _child, _name| {
                        async move { Ok("done".into()) }.boxed()
                    },
                )
                .now_or_never();
            release_first.send(()).expect("first writer is waiting");
            let second = second
                .expect("disjoint writer must not wait for the active writer")
                .expect("disjoint writer starts");
            assert_eq!(second.status, SubagentStatus::Completed);
            assert_eq!(
                first
                    .await
                    .expect("join first")
                    .expect("first summary")
                    .status,
                SubagentStatus::Completed
            );
            let _ = std::fs::remove_dir_all(left);
            let _ = std::fs::remove_dir_all(right);
        }
    }

    #[tokio::test]
    async fn cancel_active_records_cancelled_and_releases_guard() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let run_runtime = runtime.clone();
        let run_barrier = Arc::clone(&barrier);
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    no_event_sender(),
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _session_transport_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            run_barrier.wait().await;
                            std::future::pending::<Result<String>>().await
                        }
                        .boxed()
                    },
                )
                .await
        });

        barrier.wait().await;
        assert!(runtime.cancel_active());

        let summary = run.await.expect("join run").expect("run summary");
        assert_eq!(summary.status, SubagentStatus::Cancelled);

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 _transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| { async move { Ok("done".into()) }.boxed() },
            )
            .await
            .expect("second run succeeds after cancellation");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn dropped_foreground_claim_releases_background_delivery() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let started = runtime
            .start_named_governed(
                &agent,
                "explorer",
                crate::agent::SubagentInvocation {
                    prompt: "inspect".into(),
                    input: test_governance().input,
                    model: None,
                    parent_tool_call_id: Some("call-claim".into()),
                },
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
            )
            .expect("start run");
        let run_id = started.run_id().to_string();
        let claim = runtime
            .claim_foreground(&run_id)
            .expect("claim succeeds")
            .expect("active run can be claimed");
        assert!(runtime.is_foregrounded(&run_id));
        drop(claim);
        assert!(!runtime.is_foregrounded(&run_id));
        drop(started);
    }

    #[tokio::test]
    async fn wait_and_cancel_address_one_run_by_id() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let mut invocation_input = test_governance().input;
        invocation_input.background = true;
        let started = runtime
            .start_named_governed(
                &agent,
                "explorer",
                crate::agent::SubagentInvocation {
                    prompt: "inspect".into(),
                    input: invocation_input,
                    model: None,
                    parent_tool_call_id: Some("call-1".into()),
                },
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
            )
            .expect("start run");
        let run_id = started.run_id().to_string();
        let run_runtime = runtime.clone();
        let run = tokio::spawn(async move { run_runtime.complete_started_run(started).await });

        assert!(
            runtime
                .active_jobs()
                .iter()
                .any(|job| job.run_id == run_id && job.active)
        );
        assert!(runtime.cancel_run(&run_id));
        let waited = runtime
            .wait_for_result(&run_id)
            .await
            .expect("wait succeeds")
            .expect("terminal result available");
        assert_eq!(waited.status, SubagentStatus::Cancelled);
        assert_eq!(
            run.await.expect("join run").expect("run summary").status,
            SubagentStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn parent_transcript_records_running_lifecycle_and_terminal_result_only() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();

        let run_summary = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect src/subagent.rs".into(),
                test_governance(),
                sessions_dir,
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 _transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| {
                    async move { Ok("completed summary".into()) }.boxed()
                },
            )
            .await
            .expect("run succeeds");

        let parent_records = read_records(parent_dir.join(format!("{}.jsonl", parent_session_id)))
            .expect("read parent records");

        assert_eq!(run_summary.status, SubagentStatus::Completed);
        assert_eq!(parent_records.len(), 3);
        match &parent_records[0].event {
            crate::transcript::TranscriptEvent::SubagentStarted {
                run_id,
                child_session_id,
                summary,
                ..
            } => {
                assert_eq!(run_id, &run_summary.run_id);
                assert_eq!(child_session_id, &run_summary.child_session_id);
                assert_eq!(summary, "inspect src/subagent.rs");
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
        match &parent_records[1].event {
            crate::transcript::TranscriptEvent::SubagentResult {
                status,
                summary,
                child_session_id,
                ..
            } => {
                assert_eq!(status, "completed");
                assert_eq!(summary, "completed summary");
                assert_eq!(child_session_id, &run_summary.child_session_id);
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
        match &parent_records[2].event {
            crate::transcript::TranscriptEvent::Evidence {
                source, summary, ..
            } => {
                assert_eq!(summary, "completed summary");
                assert!(matches!(
                    source,
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        ..
                    } if run_id == &run_summary.run_id && child_session_id == &run_summary.child_session_id
                ));
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn large_structured_result_records_parent_evidence_in_full() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();

        let long_summary = "s".repeat(9_000);
        let report = serde_json::json!({
            "status": "completed",
            "summary": long_summary,
            "findings": ["inspection complete"],
        })
        .to_string();

        let run_summary = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect src/subagent.rs".into(),
                test_governance(),
                sessions_dir,
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                no_event_sender(),
                None,
                move |_agent, _task, _transcript, _tx, _child, _name| {
                    async move { Ok(report) }.boxed()
                },
            )
            .await
            .expect("run succeeds");

        assert_eq!(run_summary.status, SubagentStatus::Completed);
        let parent_records = read_records(parent_dir.join(format!("{parent_session_id}.jsonl")))
            .expect("read parent records");
        let evidence_detail = parent_records
            .iter()
            .find_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::Evidence { detail, .. } => detail.clone(),
                _ => None,
            })
            .expect("parent evidence carries the structured result");
        assert!(evidence_detail.contains(&long_summary));
    }

    struct PartialFailParentSink {
        file: std::fs::File,
    }

    struct MissingDelimiterParentSink {
        file: std::fs::File,
    }

    impl JournalSink for MissingDelimiterParentSink {
        fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            let partial_len = bytes.len().saturating_sub(1);
            self.file.write_all(&bytes[..partial_len])?;
            Err(io::Error::other("injected missing delimiter failure"))
        }

        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }

        fn sync_data(&mut self) -> io::Result<()> {
            self.file.sync_data()
        }
    }

    impl JournalSink for PartialFailParentSink {
        fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            let partial_len = (bytes.len() / 2).max(1);
            self.file.write_all(&bytes[..partial_len])?;
            Err(io::Error::other("injected partial parent start failure"))
        }

        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }

        fn sync_data(&mut self) -> io::Result<()> {
            self.file.sync_data()
        }
    }

    struct FailingParentSink {
        fail: Arc<AtomicBool>,
    }

    impl JournalSink for FailingParentSink {
        fn write_all(&mut self, _bytes: &[u8]) -> io::Result<()> {
            if self.fail.load(Ordering::SeqCst) {
                Err(io::Error::other("injected parent result failure"))
            } else {
                Ok(())
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }

        fn sync_data(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn partial_parent_start_failure_repairs_tail_and_publishes_durable_terminal_job() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let (parent_session_id, parent_path) = {
            let recorder = parent_recorder.lock().expect("lock parent recorder");
            (
                recorder.session_id().to_string(),
                recorder.path().to_path_buf(),
            )
        };
        let partial_file = std::fs::OpenOptions::new()
            .append(true)
            .open(&parent_path)
            .expect("open parent transcript for partial failure");
        parent_recorder
            .lock()
            .expect("lock parent recorder")
            .replace_sink_for_test(Box::new(PartialFailParentSink { file: partial_file }));

        let error = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect".into(),
                test_governance(),
                sessions_dir.clone(),
                parent_session_id.clone(),
                "turn-1".into(),
                Some(parent_recorder),
                no_event_sender(),
                None,
                |_agent, _task, _transcript, _tx, _child, _name| {
                    async move { Ok("must not run".into()) }.boxed()
                },
            )
            .await
            .expect_err("parent start write fails");
        assert!(error.to_string().contains("partial parent start failure"));

        let records = read_records(&parent_path).expect("repaired parent transcript is readable");
        let run_id = records
            .iter()
            .find_map(|record| match &record.event {
                crate::transcript::TranscriptEvent::SubagentStarted { run_id, .. } => {
                    Some(run_id.clone())
                }
                _ => None,
            })
            .expect("recovery writes parent ownership");
        assert!(records.iter().any(|record| matches!(
            &record.event,
            crate::transcript::TranscriptEvent::SubagentResult {
                run_id: recorded,
                status,
                ..
            } if recorded == &run_id && status == "failed"
        )));
        let jobs = crate::transcript::project_subagent_jobs(&sessions_dir, &records)
            .expect("project repaired terminal job");
        assert!(
            jobs.iter()
                .any(|job| { job.run_id == run_id && !job.active && job.status == "failed" }),
            "jobs={jobs:?} records={records:?}"
        );
        assert_eq!(
            runtime
                .completed_result(&run_id)
                .expect("runtime terminal result")
                .status,
            SubagentStatus::Failed
        );
    }

    #[tokio::test]
    async fn missing_parent_append_delimiter_is_repaired_before_terminal_job_publish() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let (parent_session_id, parent_path) = {
            let recorder = parent_recorder.lock().expect("lock parent recorder");
            (
                recorder.session_id().to_string(),
                recorder.path().to_path_buf(),
            )
        };
        let partial_file = std::fs::OpenOptions::new()
            .append(true)
            .open(&parent_path)
            .expect("open parent transcript for delimiter failure");
        parent_recorder
            .lock()
            .expect("lock parent recorder")
            .replace_sink_for_test(Box::new(MissingDelimiterParentSink { file: partial_file }));

        let error = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect".into(),
                test_governance(),
                sessions_dir,
                parent_session_id,
                "turn-1".into(),
                Some(parent_recorder),
                no_event_sender(),
                None,
                |_agent, _task, _transcript, _tx, _child, _name| {
                    async move { Ok("must not run".into()) }.boxed()
                },
            )
            .await
            .expect_err("parent start write fails after the complete JSON payload");
        assert!(error.to_string().contains("missing delimiter failure"));

        let records = read_records(&parent_path).expect("repaired transcript stays valid JSONL");
        assert_eq!(records.len(), 4);
        assert!(matches!(
            records[0].event,
            crate::transcript::TranscriptEvent::SubagentStarted { .. }
        ));
        assert!(matches!(
            records[1].event,
            crate::transcript::TranscriptEvent::SubagentStarted { .. }
        ));
        assert!(matches!(
            records[2].event,
            crate::transcript::TranscriptEvent::SubagentResult { .. }
        ));
        assert!(matches!(
            records[3].event,
            crate::transcript::TranscriptEvent::Evidence { .. }
        ));
    }

    #[tokio::test]
    async fn parent_result_failure_publishes_one_failed_terminal_summary() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();
        let fail_parent_result = Arc::new(AtomicBool::new(false));
        parent_recorder
            .lock()
            .expect("lock parent recorder")
            .replace_sink_for_test(Box::new(FailingParentSink {
                fail: Arc::clone(&fail_parent_result),
            }));
        let barrier = Arc::new(Barrier::new(2));
        let (release_executor, wait_for_release) = oneshot::channel();
        let exec_barrier = Arc::clone(&barrier);
        let run_runtime = runtime.clone();
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    parent_session_id,
                    "turn-1".into(),
                    Some(parent_recorder),
                    no_event_sender(),
                    None,
                    move |_agent, _task, _transcript, _tx, _child, _name| {
                        async move {
                            exec_barrier.wait().await;
                            wait_for_release.await.expect("release executor");
                            Ok("completed summary".into())
                        }
                        .boxed()
                    },
                )
                .await
        });
        barrier.wait().await;
        fail_parent_result.store(true, Ordering::SeqCst);
        release_executor.send(()).expect("executor is waiting");
        let summary = run
            .await
            .expect("join run")
            .expect("terminal summary remains available");

        assert_eq!(summary.status, SubagentStatus::Failed);
        assert_eq!(summary.failure_kind, Some(SubagentFailureKind::Hard));
        assert!(
            summary
                .summary
                .contains("failed to record parent subagent result")
        );
        let completed = runtime
            .completed_result(&summary.run_id)
            .expect("terminal result retained");
        assert_eq!(completed.status, SubagentStatus::Failed);
        assert_eq!(completed.summary, summary.summary);
        let waited = runtime
            .wait_for_result(&summary.run_id)
            .await
            .expect("wait succeeds")
            .expect("wait returns terminal result");
        assert_eq!(waited, summary);
    }

    #[tokio::test]
    async fn dropped_run_future_releases_concurrency_guard() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let run_runtime = runtime.clone();
        let run_barrier = Arc::clone(&barrier);
        let run_sessions_dir = sessions_dir.clone();
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    run_sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    no_event_sender(),
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _session_transport_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            run_barrier.wait().await;
                            std::future::pending::<Result<String>>().await
                        }
                        .boxed()
                    },
                )
                .await
        });

        barrier.wait().await;
        let run_id = runtime
            .active_jobs()
            .into_iter()
            .find(|job| job.agent_name == "explorer")
            .expect("aborted run is active")
            .run_id;
        run.abort();
        assert!(
            run.await
                .expect_err("run task should be aborted")
                .is_cancelled()
        );
        let abandoned = runtime
            .completed_result(&run_id)
            .expect("aborted run publishes a terminal result");
        assert_eq!(abandoned.status, SubagentStatus::Cancelled);
        let child_records = read_records(
            crate::transcript::child_sessions_dir(&sessions_dir)
                .join(format!("{}.jsonl", abandoned.child_session_id)),
        )
        .expect("read abandoned child transcript");
        assert!(child_records.iter().any(|record| matches!(
            &record.event,
            crate::transcript::TranscriptEvent::SubagentLifecycle { status, .. }
                if status == "cancelled"
        )));

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                no_event_sender(),
                None,
                |_agent,
                 _task,
                 _transcript,
                 _session_transport_tx,
                 _child_session_id,
                 _agent_name| { async move { Ok("done".into()) }.boxed() },
            )
            .await
            .expect("second run succeeds after aborted caller");
        assert_eq!(next.status, SubagentStatus::Completed);
    }
}
