use super::*;
use crate::session::{
    AssistantDeltaEvent, ReasoningDeltaEvent, TodoSnapshotEvent, TokenUsageEvent,
    ToolFinishedEvent, ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent,
};
use crate::tool::ToolOutputStream;
use agent_client_protocol::schema::MaybeUndefined;

fn usage(used: u64, size: u64) -> TokenUsageEvent {
    TokenUsageEvent {
        used_tokens: used,
        context_window_tokens: size,
        input_tokens: 0,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: Vec::new(),
    }
}

fn chunk_text(chunk: &ContentChunk) -> &str {
    match &chunk.content {
        ContentBlock::Text(text) => &text.text,
        other => panic!("expected text content, got {other:?}"),
    }
}

fn tool_content_text(update: &ToolCallUpdate) -> String {
    let chunks = update.fields.content.clone().expect("content field");
    chunks
        .into_iter()
        .map(|chunk| match chunk {
            ToolCallContent::Content(content) => match content.content {
                ContentBlock::Text(text) => text.text,
                other => panic!("expected text content, got {other:?}"),
            },
            other => panic!("expected content tool call content, got {other:?}"),
        })
        .collect::<String>()
}

#[test]
fn assistant_delta_projects_to_agent_message_chunk() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::AssistantDelta(
        AssistantDeltaEvent::with_message_id("m1", "hello"),
    ));
    let Some(SessionUpdate::AgentMessageChunk(chunk)) = update else {
        panic!("expected an agent message chunk");
    };
    assert_eq!(chunk_text(&chunk), "hello");
    assert_eq!(
        chunk.message_id.map(|id| id.0.to_string()),
        Some("m1".to_string())
    );
}

#[test]
fn reasoning_delta_projects_to_agent_thought_chunk() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::ReasoningDelta(
        ReasoningDeltaEvent::new("r1", "thinking"),
    ));
    let Some(SessionUpdate::AgentThoughtChunk(chunk)) = update else {
        panic!("expected an agent thought chunk");
    };
    assert_eq!(chunk_text(&chunk), "thinking");
}

#[test]
fn user_message_projects_to_user_message_chunk() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::UserMessage(
        crate::session::UserMessageEvent::new("do the thing"),
    ));
    let Some(SessionUpdate::UserMessageChunk(chunk)) = update else {
        panic!("expected a user message chunk");
    };
    assert_eq!(chunk_text(&chunk), "do the thing");
}

#[test]
fn tool_pending_projects_to_tool_call_with_kind() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::ToolPending(ToolPendingEvent::new(
        "call-1",
        tool_names::TOOL_SHELL_EXEC,
    )));
    let Some(SessionUpdate::ToolCall(call)) = update else {
        panic!("expected a tool call");
    };
    assert_eq!(call.tool_call_id.0.to_string(), "call-1");
    assert_eq!(call.title, tool_names::TOOL_SHELL_EXEC);
    assert_eq!(call.kind, ToolKind::Execute);
    assert_eq!(call.status, ToolCallStatus::Pending);
}

#[test]
fn tool_started_projects_to_in_progress_update_with_arguments() {
    let mut projection = AcpUpdateProjection::new();
    let mut started = ToolStartedEvent::new("call-1", tool_names::TOOL_FS_READ, "Read main.rs");
    started.arguments = Some(r#"{"path":"src/main.rs"}"#.to_string());
    let update = projection.project(&SessionTransportEvent::ToolStarted(started));
    let Some(SessionUpdate::ToolCallUpdate(update)) = update else {
        panic!("expected a tool call update");
    };
    assert_eq!(update.fields.status, Some(ToolCallStatus::InProgress));
    assert_eq!(update.fields.title.as_deref(), Some("Read main.rs"));
    assert_eq!(update.fields.kind, Some(ToolKind::Read));
    assert_eq!(
        update.fields.raw_input,
        Some(serde_json::json!({"path": "src/main.rs"}))
    );
}

#[test]
fn tool_output_deltas_accumulate_into_replaced_content() {
    let mut projection = AcpUpdateProjection::new();
    let first = projection.project(&SessionTransportEvent::ToolOutputDelta(
        ToolOutputDeltaEvent::new("call-1", ToolOutputStream::Stdout, "line 1\n"),
    ));
    let Some(SessionUpdate::ToolCallUpdate(first)) = first else {
        panic!("expected a tool call update");
    };
    assert_eq!(tool_content_text(&first), "line 1\n");

    let second = projection.project(&SessionTransportEvent::ToolOutputDelta(
        ToolOutputDeltaEvent::new("call-1", ToolOutputStream::Stdout, "line 2\n"),
    ));
    let Some(SessionUpdate::ToolCallUpdate(second)) = second else {
        panic!("expected a tool call update");
    };
    assert_eq!(tool_content_text(&second), "line 1\nline 2\n");
}

#[test]
fn tool_finished_reports_terminal_status_and_output() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-1".to_string(),
        name: tool_names::TOOL_FS_READ.to_string(),
        summary: "Read main.rs".to_string(),
        outcome: ToolOutcome::Failure,
        output: Some("permission denied".to_string()),
    }));
    let Some(SessionUpdate::ToolCallUpdate(update)) = update else {
        panic!("expected a tool call update");
    };
    assert_eq!(update.fields.status, Some(ToolCallStatus::Failed));
    assert_eq!(tool_content_text(&update), "permission denied");
}

#[test]
fn token_usage_projects_to_usage_update_only_with_a_context_window() {
    let mut projection = AcpUpdateProjection::new();
    assert!(
        projection
            .project(&SessionTransportEvent::SessionTokenUsage(usage(0, 0)))
            .is_none()
    );

    let update = projection.project(&SessionTransportEvent::TokenUsage(usage(120, 800)));
    let Some(SessionUpdate::UsageUpdate(update)) = update else {
        panic!("expected a usage update");
    };
    assert_eq!(update.used, 120);
    assert_eq!(update.size, 800);
}

#[test]
fn session_title_projects_to_session_info_update() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::SessionTitleUpdated {
        session_id: "session-1".to_string(),
        title: "Fix the parser".to_string(),
    });
    let Some(SessionUpdate::SessionInfoUpdate(update)) = update else {
        panic!("expected a session info update");
    };
    assert_eq!(
        update.title,
        MaybeUndefined::Value("Fix the parser".to_string())
    );
}

fn todo(id: &str, content: &str, status: TodoStatus) -> TodoItem {
    TodoItem {
        id: id.to_string(),
        content: content.to_string(),
        status,
    }
}

/// The entry fields a test compares: an ACP entry states its content,
/// priority, and status, and nothing else the todo model tracks.
fn entry_fields(entry: &PlanEntry) -> (String, PlanEntryPriority, PlanEntryStatus) {
    (
        entry.content.clone(),
        entry.priority.clone(),
        entry.status.clone(),
    )
}

#[test]
fn todo_snapshot_projects_to_a_plan_carrying_every_item() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::TodoSnapshot(
        TodoSnapshotEvent::new(vec![
            todo("open", "wire parser", TodoStatus::Pending),
            todo("active", "port tests", TodoStatus::InProgress),
            todo("waiting", "await review", TodoStatus::Blocked),
            todo("finished", "scaffold module", TodoStatus::Completed),
            todo("dropped", "abandoned spike", TodoStatus::Cancelled),
        ]),
    ));
    let Some(SessionUpdate::Plan(plan)) = update else {
        panic!("expected a plan");
    };
    assert_eq!(
        plan.entries.iter().map(entry_fields).collect::<Vec<_>>(),
        vec![
            (
                "wire parser".to_string(),
                PlanEntryPriority::Medium,
                PlanEntryStatus::Pending
            ),
            (
                "port tests".to_string(),
                PlanEntryPriority::Medium,
                PlanEntryStatus::InProgress
            ),
            (
                "await review".to_string(),
                PlanEntryPriority::Medium,
                PlanEntryStatus::Pending
            ),
            (
                "scaffold module".to_string(),
                PlanEntryPriority::Medium,
                PlanEntryStatus::Completed
            ),
            (
                "abandoned spike".to_string(),
                PlanEntryPriority::Medium,
                PlanEntryStatus::Completed
            ),
        ]
    );
}

#[test]
fn an_empty_todo_snapshot_clears_the_plan() {
    let mut projection = AcpUpdateProjection::new();
    let update = projection.project(&SessionTransportEvent::TodoSnapshot(
        TodoSnapshotEvent::new(Vec::new()),
    ));
    let Some(SessionUpdate::Plan(plan)) = update else {
        panic!("expected a plan");
    };
    assert!(plan.entries.is_empty());
}

#[test]
fn events_without_an_acp_counterpart_project_to_nothing() {
    let mut projection = AcpUpdateProjection::new();
    assert!(
        projection
            .project(&SessionTransportEvent::ModelChanged {
                model_id: "provider/model".to_string(),
            })
            .is_none()
    );
    assert!(projection.project(&SessionTransportEvent::Done).is_none());
}
