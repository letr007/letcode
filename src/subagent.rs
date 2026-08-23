mod pool;
mod result;
mod route_factory;

pub use pool::{SubagentPool, SubagentRunGovernance};
pub use result::{
    StructuredSubagentResult, SubagentFailureKind, SubagentRunSummary, SubagentStatus,
    looks_like_structured_subagent_output, try_parse_structured_subagent_result,
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
use async_openai::Client;
#[cfg(test)]
use async_openai::config::OpenAIConfig;
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
use tokio::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiProtocol;
    use crate::session::SessionTransportEvent;
    use crate::transcript::read_records;
    use async_openai::Client;
    use async_openai::config::OpenAIConfig;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::sync::Barrier;
    use tokio::time::sleep;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(Client::with_config(OpenAIConfig::new()), "gpt-test", 2, 4)
    }

    fn temp_sessions_dir() -> PathBuf {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("{}-{id}", generate_run_id()))
    }

    fn no_event_sender() -> Option<crate::subagent_events::SubagentEventSender<OpenAIConfig>> {
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

    async fn wait_until<F>(mut condition: F)
    where
        F: FnMut() -> bool,
    {
        for _ in 0..50 {
            if condition() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        assert!(condition(), "condition was not met before timeout");
    }

    #[test]
    fn child_agents_do_not_expose_recursive_subagent_tools() {
        let agent = test_agent();
        let child = AgentFactory::create_child(&agent, &AgentTemplate::fixer());
        let tool_names = child
            .tool_definitions_for_test()
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();

        assert!(!tool_names.iter().any(|name| name == "agent__explore"));
        assert!(!tool_names.iter().any(|name| name == "agent__fixer"));
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
            protocol,
            context_window: None,
            effective_input_limit_tokens: None,
            max_output_tokens: None,
            supports_tools: false,
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
            api_key: api_key.into(),
            protocol,
            default_model: models.first().copied().unwrap_or_default().into(),
            retry: None,
            models: models
                .iter()
                .map(|model| ((*model).to_string(), test_model_config(protocol)))
                .collect(),
        }
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
        let factory = ExpertRouteFactory::new_with_policies(
            [("explorer".into(), None, Vec::new())],
            &providers,
            &RetryConfig::default(),
        )
        .expect("factory should build");
        let mut parent = test_agent();
        parent.set_primary_route(ModelRoute::new("primary", "shared"));

        let inherited = ModelRoute::new("primary", "shared");
        assert_eq!(
            SubagentChildFactory::resolve_route(
                &factory,
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
                &factory,
                &parent,
                &AgentTemplate::explorer(),
                Some(&inherited),
                true,
            )
            .expect("takeover may reuse the effective default route"),
            inherited
        );
        let error = SubagentChildFactory::resolve_route(
            &factory,
            &parent,
            &AgentTemplate::explorer(),
            Some(&ModelRoute::new("primary", "shared")),
            false,
        )
        .expect_err("empty allowlist rejects explicit selection");
        assert!(
            error
                .to_string()
                .contains("is not allowed for expert 'explorer'")
        );
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
        let factory = ExpertRouteFactory::new_with_policies(
            [(
                "explorer".into(),
                Some(ModelRoute::new("primary", "shared")),
                vec![selected.clone()],
            )],
            &providers,
            &RetryConfig::default(),
        )
        .expect("factory should build");
        let mut parent = test_agent();
        parent.set_primary_route(ModelRoute::new("primary", "shared"));

        for takeover in [false, true] {
            assert_eq!(
                SubagentChildFactory::resolve_route(
                    &factory,
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
            &factory,
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
    fn expert_route_factory_rejects_an_unconfigured_model_for_a_known_provider() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
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

        assert!(matches!(
            result,
            Err(error)
                if error.to_string()
                    == "child route provider 'expert' model 'unconfigured' is not configured"
        ));
    }

    #[test]
    fn routed_child_retains_route_factory_for_takeover_restoration() {
        let provider = ProviderConfig {
            base_url: "http://127.0.0.1:9876/v1".into(),
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
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
            .expect("factory should build"),
        );
        let mut parent = test_agent();
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
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: Some(RetryConfig {
                enabled: false,
                max_attempts: 1,
                max_recovery_attempts: 1,
                initial_delay_secs: 1,
                backoff_multiplier: 1.0,
                jitter_secs: 0,
            }),
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    context_window: Some(8_192),
                    effective_input_limit_tokens: Some(4_096),
                    max_output_tokens: Some(512),
                    supports_tools: false,
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
        let factory = ExpertRouteFactory::new(
            [("explorer".into(), ModelRoute::new("expert", "shared"))],
            &providers,
            &RetryConfig::default(),
        )
        .expect("factory should build");
        let mut parent = test_agent();
        parent.set_subagent_child_factory(Arc::new(factory));

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
    fn structured_result_parser_preserves_object_shaped_validation_outcomes() {
        let result = StructuredSubagentResult::from_model_output(
            r#"{"status":"completed","summary":"done","validation":[{"command":"cargo test","result":"failed"},{"command":"cargo fmt","result":"not_run"}]}"#,
            SubagentStatus::Completed,
            "run-1",
            "child-1",
        );

        assert_eq!(
            result.validation,
            vec!["cargo test failed", "cargo fmt not_run"]
        );
    }

    #[test]
    fn runtime_failures_are_classified_hard_and_model_failures_logical() {
        let hard = build_runtime_summary(
            "run-1",
            "child-1",
            "fixer",
            SubagentStatus::Failed,
            "provider connection failed".into(),
        );
        let logical = build_completed_summary(
            "run-2",
            "child-2",
            "fixer",
            r#"{"status":"failed","summary":"task requirements not met"}"#.into(),
        );

        assert_eq!(hard.failure_kind, Some(SubagentFailureKind::Hard));
        assert_eq!(logical.failure_kind, Some(SubagentFailureKind::Logical));
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

    #[tokio::test]
    async fn fixer_out_of_scope_changes_are_visible_and_fail_the_run() {
        let runtime = SubagentPool::new();
        let owned = temp_scope_root("owned");
        let owned_label = owned.to_string_lossy().into_owned();
        let mut governance = test_governance();
        governance.input.allowed_paths = vec![owned_label.clone()];
        governance.input.owned_paths = vec![owned_label];

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
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
                            r#"{"status":"completed","summary":"changed files","files_changed":["src/outside.rs"]}"#
                                .into(),
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

    #[tokio::test]
    async fn observed_child_write_effects_enforce_scope_even_when_files_changed_missing() {
        let runtime = SubagentPool::new();
        let owned = temp_scope_root("allowed");
        let mut governance = test_governance();
        governance.input.allowed_paths = vec![owned.to_string_lossy().into_owned()];

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
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

    #[tokio::test]
    async fn takeover_restores_full_runtime_snapshot_before_appending_prompt() {
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
        let first = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect initial state".into(),
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
                        let mut transcript = transcript.lock().expect("lock child transcript");
                        transcript.record_user_message("initial prompt")?;
                        transcript.record_model_changed("gpt-test", "test/child-resume-model")?;
                        transcript.record_assistant_tool_call_batch(
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
                [("explorer".into(), None, vec![takeover_route.clone()])],
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
            .expect("takeover route factory"),
        );
        let mut takeover_parent = test_agent();
        takeover_parent.set_primary_route(takeover_route);
        takeover_parent.set_subagent_child_factory(factory);
        let mut takeover_governance = test_governance();
        takeover_governance.input.target_child_session_id = Some(first.child_session_id.clone());
        let resumed_child_session_id = first.child_session_id.clone();
        let resumed = runtime
            .run_with_executor(
                &takeover_parent,
                AgentTemplate::explorer(),
                "continue inspection".into(),
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
                        assert_eq!(agent.model(), "child-resume-model");
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
            api_key: "expert-key".into(),
            protocol: ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: ApiProtocol::Completions,
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
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
                )],
                &indexmap::IndexMap::from([("expert".into(), provider)]),
                &RetryConfig::default(),
            )
            .expect("factory should build"),
        );
        let mut parent = test_agent();
        parent.set_primary_route(ModelRoute::new("primary", "shared"));
        parent.set_subagent_child_factory(factory.clone());
        parent.set_primary_route_factory(factory.clone());
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
    async fn max_concurrency_guard_rejects_second_run() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

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
                            tokio::time::sleep(Duration::from_millis(50)).await;
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
            .await
            .expect_err("second run should be rejected");

        let error = second.to_string();
        assert!(
            error.contains("is busy") && error.contains("explorer"),
            "{error}"
        );
        assert!(error.contains("only one active run per role"), "{error}");
        assert!(error.contains("run_id="), "{error}");
        assert!(error.contains("child_session_id="), "{error}");
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
                pool_ordinal: _,
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
    async fn dropped_run_future_releases_concurrency_guard() {
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
        run.abort();
        assert!(
            run.await
                .expect_err("run task should be aborted")
                .is_cancelled()
        );

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
