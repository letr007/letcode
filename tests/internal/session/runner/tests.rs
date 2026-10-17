use super::formatting::summarize_search;
use super::*;
use crate::agent::{AutoContinueState, CacheUsageReport, SubagentInvocation, TodoItem, TodoStatus};
use crate::permission::PermissionRequest;
use crate::session::{PermissionDecision, SessionEvent};
use crate::transcript::TranscriptRecorder;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn model_failure_detail_reaches_error_event_and_recorded_message() {
    let error = anyhow::Error::new(
        crate::model_runtime::ModelFailure::new(
            crate::model_runtime::FailurePhase::Transport,
            crate::model_runtime::FailureKind::Http,
        )
        .with_status(400)
        .with_detail("provider says the request field is invalid"),
    );
    let event = error_event(&error);
    assert!(event.message.contains("status 400"));
    assert_eq!(
        event.details.as_deref(),
        Some("provider says the request field is invalid")
    );
    assert!(recorded_error_message(&event).contains("Provider response:"));
}

#[test]
fn summarize_search_uses_aggregate_and_marks_folded() {
    assert_eq!(
        summarize_search(&json!({
            "matches": [],
            "total_matches": 843,
            "files": 12,
            "folded": true,
        })),
        "843 matches · 12 files · folded"
    );
    // Legacy payloads without aggregates fall back to the inline array.
    assert_eq!(
        summarize_search(&json!({
            "matches": [{"path": "a"}, {"path": "b"}],
        })),
        "2 matches"
    );
}

#[tokio::test]
async fn permission_request_handle_delivers_approval() {
    let (tx, rx) = oneshot::channel();
    let handle = RunnerPermissionRequest::new(tx);

    handle.approve().expect("approve succeeds");

    assert_eq!(
        rx.await.expect("receiver gets response"),
        PermissionResponse::AllowOnce
    );
}

fn credential_delegate(
    parent: &mut Agent,
    default_route: crate::config::ModelRoute,
    allowed_models: Vec<crate::config::ModelRoute>,
) -> RunnerSubagentDelegate {
    let provider = crate::config::ProviderConfig {
        base_url: "http://127.0.0.1:9/v1".into(),
        jev_endpoint: None,
        auth_mode: crate::config::ProviderAuthMode::ApiKey,
        api_key: "expert-key".into(),
        protocol: crate::config::ApiProtocol::Completions,
        default_model: "shared".into(),
        retry: None,
        reviewer: None,
        models: indexmap::IndexMap::from([(
            "shared".into(),
            crate::config::ModelConfig {
                display_name: None,
                protocol: crate::config::ApiProtocol::Completions,
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
    let factory = crate::subagent::ExpertRouteFactory::new_with_policies(
        [("explorer".into(), Some(default_route), allowed_models, None)],
        &indexmap::IndexMap::from([("expert".into(), provider)]),
        &crate::config::RetryConfig::default(),
    )
    .expect("credential test factory");
    parent.set_subagent_child_factory(Arc::new(factory));
    RunnerSubagentDelegate {
        child_completions: None,
        runtime: SubagentPool::new(),
        sessions_dir: std::env::temp_dir(),
        transcript: temp_transcript(),
        event_tx: None,
        route_api_key_configured: indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]),
        provider_api_key_hints: indexmap::IndexMap::from([(
            "expert".into(),
            "Set EXPERT_API_KEY.".into(),
        )]),
        api_key_hint: "Set <PROVIDER>_API_KEY.".into(),
        background_control_tx: None,
        background_event_tx: None,
        background_child_started_tx: None,
        child_live_text: None,
    }
}

fn test_evidence() -> crate::evidence::EvidenceRecord {
    crate::evidence::EvidenceRecord {
        id: "evidence-1".into(),
        sequence: 1,
        timestamp_ms: 0,
        evidence_kind: crate::evidence::EvidenceKind::CommandResult,
        title: "command result".into(),
        summary: "command completed".into(),
        detail: None,
        source: crate::evidence::EvidenceSource::Command {
            command: "cargo check".into(),
            status: Some(0),
        },
        tags: Vec::new(),
    }
}

#[test]
fn tool_batch_reconciliation_returns_actions_at_evidence_and_batch_boundaries() {
    let mut reconciliation = ToolBatchReconciliation::default();
    let output = crate::tool::ToolResult::ok("shell__exec", json!({}));
    let actions: Vec<_> = [
        (
            AgentEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output,
            },
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: false,
            },
        ),
        (
            AgentEvent::EvidenceRecorded(test_evidence()),
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: false,
            },
        ),
        (
            AgentEvent::ToolCallBatchFinished,
            JournalEffect {
                persisted: false,
                context_projection: ContextProjection::None,
                compaction_terminal: false,
            },
        ),
        (
            AgentEvent::EvidenceRecorded(test_evidence()),
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: false,
            },
        ),
    ]
    .into_iter()
    .map(|(event, effect)| reconciliation.reconcile(&event, effect))
    .collect();

    assert_eq!(
        actions,
        vec![
            ToolBatchReconciliationAction::None,
            ToolBatchReconciliationAction::None,
            ToolBatchReconciliationAction::FinishBatch(RuntimeContextDisposition::Advance),
            ToolBatchReconciliationAction::ProjectEvidence,
        ]
    );
}

#[test]
fn tool_batch_reconciliation_defers_projection_until_batch_boundary() {
    let mut reconciliation = ToolBatchReconciliation::default();
    reconciliation.observe_event(&AgentEvent::ToolCallFinished {
        call_id: "call-1".into(),
        name: "shell__exec".into(),
        ok: true,
        output: crate::tool::ToolResult::ok("shell__exec", json!({})),
    });

    assert!(!reconciliation.should_project_evidence());

    let disposition = reconciliation.finish();
    assert_eq!(disposition, RuntimeContextDisposition::Advance);
    assert!(reconciliation.should_project_evidence());
}

#[test]
fn tool_batch_reconciliation_keeps_non_tool_evidence_on_its_own_boundary() {
    let mut reconciliation = ToolBatchReconciliation::default();
    assert!(reconciliation.should_project_evidence());

    reconciliation.record_journal_effect(JournalEffect {
        persisted: true,
        context_projection: ContextProjection::ReplaceScope,
        compaction_terminal: false,
    });
    assert_eq!(
        reconciliation.finish(),
        RuntimeContextDisposition::ReplaceScope
    );
    assert!(reconciliation.should_project_evidence());
}

#[test]
fn permission_request_event_carries_subagent_origin() {
    let request = PermissionRequest {
        call_id: Some("call-8".into()),
        tool: "shell__exec".into(),
        args: json!({"command": "cargo test"}),
        class: crate::permission::ToolPermissionClass::Command,
        summary: "shell__exec cargo test".into(),
        preview: None,
        can_allow_always: false,
        grant_summary: None,
    };

    let event = permission_request_event(&request, Some("fixer"));

    assert_eq!(event.call_id, "call-8");
    assert_eq!(event.tool_name, "shell__exec");
    assert_eq!(event.origin_label.as_deref(), Some("fixer"));
}

fn temp_transcript() -> Arc<Mutex<TranscriptRecorder>> {
    let base_dir = std::env::temp_dir().join(format!(
        "letcode-runner-child-streaming-test-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time ok")
            .as_nanos()
    ));
    std::fs::create_dir_all(&base_dir).expect("temp dir created");
    Arc::new(Mutex::new(
        TranscriptRecorder::create(&base_dir).expect("transcript created"),
    ))
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn fast_mode_auto_disable_keeps_child_settings_out_of_parent_events() {
        use crate::agent::{AgentFactory, AgentTemplate};
        use crate::config::ModelRoute;
        use crate::model_runtime::{ProtocolRegistry, RuntimeConfig};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{Duration, timeout};

        for child_session_id in [None, Some("child-session")] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                timeout(Duration::from_secs(5), async move {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut request = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        let read = stream.read(&mut chunk).await.unwrap();
                        assert!(read > 0);
                        request.extend_from_slice(&chunk[..read]);
                        if let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                            let headers = std::str::from_utf8(&request[..at]).unwrap();
                            let length = headers.lines().find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            }).unwrap();
                            if request.len() >= at + 4 + length {
                                break;
                            }
                        }
                    }
                    let body = concat!(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
                        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\"}}\n\n"
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
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
            let route = Arc::new(catalog.route("test", "model").unwrap().clone());
            let mode = crate::fast_mode::FastMode::load("letcode.toml", true);
            let mut parent = Agent::new("gpt-test", 1, 1);
            parent.set_fast_mode(mode.clone());
            let mut agent = if child_session_id.is_some() {
                AgentFactory::create_child(&parent, &AgentTemplate::explorer())
            } else {
                parent
            };
            agent.set_resolved_runtime_catalog(Some(catalog));
            agent.set_model_route_authority(ModelRoute::new("test", "model"), route);
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let mut runner = AgentRunner::new(tx);
            runner.child_session_id = child_session_id.map(str::to_owned);
            let result = timeout(
                Duration::from_secs(5),
                runner.run_prompt(
                    &mut agent,
                    UserMessageSubmission::new("prompt", UserMessageContent::from("say ok")),
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result, "ok");
            server.await.unwrap();
            assert!(!agent.fast_mode_enabled());
            assert_eq!(mode.enabled(), child_session_id.is_some());

            let mut settings_changes = 0;
            let mut auto_disable_notice = false;
            while let Ok(event) = rx.try_recv() {
                match event {
                    SessionTransportEvent::FastModeChanged { enabled } => {
                        assert!(!enabled);
                        settings_changes += 1;
                    }
                    SessionTransportEvent::ChildSessionEvent {
                        child_session_id: id,
                        event: SessionEvent::Notice(notice),
                        ..
                    } if notice.message.contains("Fast mode auto-disabled") => {
                        assert_eq!(Some(id.as_str()), child_session_id);
                        auto_disable_notice = true;
                    }
                    SessionTransportEvent::Notice(notice)
                        if notice.message.contains("Fast mode auto-disabled") =>
                    {
                        assert!(child_session_id.is_none());
                        auto_disable_notice = true;
                    }
                    _ => {}
                }
            }
            assert_eq!(settings_changes, usize::from(child_session_id.is_none()));
            assert!(auto_disable_notice);
        }
    }

    #[test]
    fn tool_driven_invalid_override_is_reported_before_credential_lookup() {
        let mut parent = Agent::new("shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let selected = crate::config::ModelRoute::new("expert", "shared");
        let delegate = credential_delegate(
            &mut parent,
            crate::config::ModelRoute::new("expert", "shared"),
            Vec::new(),
        );
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: Some(selected.display_name()),
                target_child_session_id: None,
                background: false,
            },
            model: Some(selected),
            prompt: "inspect route credentials".into(),
            parent_tool_call_id: Some("call-invalid".into()),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create runtime");
        let output = runtime
            .block_on(delegate.run_named(&parent, "explorer", invocation))
            .expect("invalid route is a tool result");
        let message = &output.error.as_ref().expect("route error").message;
        assert!(message.contains("requested model route 'expert/shared' is not allowed"));
        assert!(!message.contains("API key is not set"));
    }

    #[test]
    fn tool_driven_expert_delegation_requires_its_route_credential() {
        let mut parent = Agent::new("shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let delegate = credential_delegate(
            &mut parent,
            crate::config::ModelRoute::new("expert", "shared"),
            Vec::new(),
        );
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
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
            model: None,
            prompt: "inspect route credentials".into(),
            parent_tool_call_id: Some("call-1".into()),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create runtime");
        let output = runtime.block_on(delegate.run_named(&parent, "explorer", invocation));

        let output = output.expect("credential denial is a tool result");
        assert!(!output.ok);
        assert!(matches!(
            output.error.as_ref(),
            Some(crate::tool::ToolError { message, .. })
                if message == "API key is not set for the selected provider. Set EXPERT_API_KEY."
        ));
        let data = output
            .data
            .expect("credential denial includes route metadata");
        assert_eq!(data.get("route"), Some(&json!("expert/shared")));
        assert_eq!(data.get("agent_name"), Some(&json!("explorer")));
    }
}
