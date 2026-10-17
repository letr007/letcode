use super::*;
use crate::model_runtime::runtime::TurnDriver;
use crate::model_runtime::{ContentPart, MessageRole, ProtocolRegistry, RuntimeConfig};
use crate::permission::PermissionApproval;

fn recovery_route() -> std::sync::Arc<crate::model_runtime::ResolvedModelRoute> {
    let catalog = RuntimeConfig::from_toml(
        r#"active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "none"
[providers.test.endpoints]
base_url = "https://test.example.invalid/v1"
[providers.test.models.model]
[providers.test.models.model.capabilities]
tools = true
parallel_tool_calls = true
[providers.test.models.model.capabilities.generation]
parallel_tool_calls = true
"#,
    )
    .unwrap()
    .resolve(&ProtocolRegistry::builtins())
    .unwrap();
    std::sync::Arc::new(catalog.route("test", "model").unwrap().clone())
}

fn plain_tool_arguments() -> serde_json::Value {
    serde_json::json!({
        "items": [{"id": "one", "content": "one", "status": "completed"}]
    })
}

fn completed_tool(id: &str, name: &str, arguments: serde_json::Value) -> CompletedToolCall {
    CompletedToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

fn resolved_driver<'a, F, E, A, Dfut, Efut, Afut>(
    agent: &'a mut Agent,
    route: std::sync::Arc<crate::model_runtime::ResolvedModelRoute>,
    on_delta: &'a mut F,
    on_event: &'a mut E,
    approve: &'a mut A,
) -> ResolvedTurnDriver<'a, F, E, A>
where
    F: FnMut(&str) -> Dfut,
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Dfut: Future<Output = Result<()>>,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    ResolvedTurnDriver {
        agent,
        route,
        protocol: ApiProtocol::Responses,
        turn_prelude: &[],
        protected_start_index: 0,
        on_delta,
        on_event,
        approve,
        tool_call_count: 0,
        user_handoff: false,
        continuation_count: 0,
        final_text: String::new(),
        prepared: None,
        usage: None,
        response_id: None,
        cache_read_tokens: None,
        cache_write_tokens: None,
        cache_details_present: false,
        current_attempt: 1,
        scheduled_retry: None,
        reasoning_summary_items: BTreeSet::new(),
        reasoning_text: BTreeMap::new(),
        fake_decorator: None,
        ws_transport: None,
        ws_previous: None,
        pending_async_tools: Vec::new(),
        async_tool_gate_open: true,
        model_tool_order: Vec::new(),
        pending_tool_done: BTreeMap::new(),
        attempt_tool_ids: BTreeSet::new(),
        recovered_tool_calls: 0,
        count_recovered_tool_calls: false,
        reconciled_async_tool_ids: BTreeSet::new(),
        attempt_assistant_persisted: false,
    }
}

#[tokio::test]
async fn ordinary_recovery_completed_call_is_cancelled_before_continuation() {
    let mut agent = Agent::new("model", 4, 4);
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.turn.turn_id = 1;
    let route = recovery_route();
    let mut on_delta = |_text: &str| async { Ok::<(), anyhow::Error>(()) };
    let mut on_event = |_event: AgentEvent| async { Ok::<(), anyhow::Error>(()) };
    let mut approve = |_request: PermissionRequest| async {
        Ok::<PermissionApproval, anyhow::Error>(PermissionApproval::AllowOnce)
    };
    let mut driver = resolved_driver(
        &mut agent,
        route,
        &mut on_delta,
        &mut on_event,
        &mut approve,
    );
    let partial = ModelAttemptSnapshot {
        assistant: ModelMessage {
            role: MessageRole::Assistant,
            content: vec![
                ContentPart::Text("partial".into()),
                ContentPart::ToolCall {
                    id: "call-plain".into(),
                    name: "workflow__todos".into(),
                    arguments: plain_tool_arguments(),
                },
            ],
        },
        completed_tools: vec![completed_tool(
            "call-plain",
            "workflow__todos",
            plain_tool_arguments(),
        )],
        events: vec![ModelEvent::ToolStarted {
            id: "call-plain".into(),
            name: "workflow__todos".into(),
        }],
        ..ModelAttemptSnapshot::default()
    };
    let failure = ModelFailure::new(
        crate::model_runtime::FailurePhase::Transport,
        crate::model_runtime::FailureKind::Http,
    );
    TurnDriver::recover_iteration(&mut driver, &partial, &failure)
        .await
        .unwrap();
    let history = driver.agent.history_for_test();
    assert!(matches!(
        history.as_slice(),
        [
            HistoryItem::AssistantTurn { text: Some(text), calls, .. },
            HistoryItem::InternalContinuation { .. },
        ] if text == "partial" && calls.is_empty()
    ));
    assert!(!history.iter().any(|item| matches!(
        item,
        HistoryItem::ToolOutput { call_id, .. } if call_id == "call-plain"
    )));
}

#[tokio::test]
async fn rejecting_an_incomplete_tool_batch_keeps_a_reasoning_only_turn_out_of_history() {
    let mut agent = Agent::new("model", 4, 4);
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.turn.turn_id = 1;
    let route = recovery_route();
    let mut on_delta = |_text: &str| async { Ok::<(), anyhow::Error>(()) };
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut on_event = {
        let events = std::sync::Arc::clone(&events);
        move |event: AgentEvent| {
            events.lock().expect("events lock").push(event);
            async { Ok::<(), anyhow::Error>(()) }
        }
    };
    let mut approve = |_request: PermissionRequest| async {
        Ok::<PermissionApproval, anyhow::Error>(PermissionApproval::AllowOnce)
    };
    let mut driver = resolved_driver(
        &mut agent,
        route,
        &mut on_delta,
        &mut on_event,
        &mut approve,
    );
    let partial = ModelAttemptSnapshot {
        assistant: ModelMessage {
            role: MessageRole::Assistant,
            content: vec![
                ContentPart::Reasoning {
                    item_id: "reasoning".into(),
                    text: "plan the exploration".into(),
                    replay: None,
                },
                ContentPart::ToolCall {
                    id: "call-explore".into(),
                    name: "workflow__todos".into(),
                    arguments: plain_tool_arguments(),
                },
            ],
        },
        completed_tools: Vec::new(),
        events: vec![ModelEvent::ToolStarted {
            id: "call-explore".into(),
            name: "workflow__todos".into(),
        }],
        ..ModelAttemptSnapshot::default()
    };
    let failure = ModelFailure::new(
        crate::model_runtime::FailurePhase::Decode,
        crate::model_runtime::FailureKind::MalformedResponse,
    )
    .with_code("invalid_completions_event");
    TurnDriver::reject_tool_batch(&mut driver, &partial, &failure)
        .await
        .unwrap();
    assert!(driver.agent.history_for_test().is_empty());
    let cancelled = events
        .lock()
        .expect("events lock")
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallCancelled { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(cancelled, vec!["call-explore".to_string()]);
}
