//! Maps session transport events onto ACP session updates.
//!
//! Only events that describe transcript, tool-call, or workflow progress are
//! projected. Interaction events carry responder handles and are answered by
//! the driver; events without an ACP counterpart are dropped here.

use std::collections::HashMap;

use agent_client_protocol::schema::v1::{
    Content, ContentBlock, ContentChunk, MessageId, Plan, PlanEntry, PlanEntryPriority,
    PlanEntryStatus, SessionInfoUpdate, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind, UsageUpdate,
};

use crate::session::{SessionTransportEvent, ToolOutcome, UserMessageEvent};
use crate::tool_names;
use crate::workflow_state::{TodoItem, TodoStatus};

/// Projects events onto ACP updates while tracking the state ACP expresses as
/// replacements rather than deltas.
#[derive(Debug, Default)]
pub(super) struct AcpUpdateProjection {
    /// Tool output streamed since the tool call started, keyed by call id.
    /// ACP replaces tool-call content instead of appending chunks, so each
    /// update carries the output accumulated so far.
    tool_output: HashMap<String, String>,
}

impl AcpUpdateProjection {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Projects an event into the single session update the ACP schema has for
    /// it.
    pub(super) fn project(&mut self, event: &SessionTransportEvent) -> Option<SessionUpdate> {
        match event {
            SessionTransportEvent::UserMessage(event) => Some(SessionUpdate::UserMessageChunk(
                message_chunk(user_message_text(event)),
            )),
            SessionTransportEvent::AssistantDelta(event) => Some(SessionUpdate::AgentMessageChunk(
                message_chunk(event.delta.clone())
                    .message_id(event.message_id.clone().map(MessageId::new)),
            )),
            SessionTransportEvent::ReasoningDelta(event) => Some(SessionUpdate::AgentThoughtChunk(
                message_chunk(event.delta.clone()),
            )),
            SessionTransportEvent::ToolPending(event) => Some(SessionUpdate::ToolCall(
                ToolCall::new(event.call_id.clone(), event.name.clone())
                    .kind(tool_kind(&event.name))
                    .status(ToolCallStatus::Pending),
            )),
            SessionTransportEvent::ToolStarted(event) => {
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    event.call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(ToolCallStatus::InProgress)
                        .title(event.summary.clone())
                        .kind(tool_kind(&event.name))
                        .raw_input(parse_arguments(event.arguments.as_deref())),
                )))
            }
            SessionTransportEvent::ToolOutputDelta(event) => {
                let output = self.tool_output.entry(event.call_id.clone()).or_default();
                output.push_str(&event.chunk);
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    event.call_id.clone(),
                    ToolCallUpdateFields::new().content(vec![text_content(output.clone())]),
                )))
            }
            SessionTransportEvent::ToolCancelled(event) => {
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    event.call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(ToolCallStatus::Failed)
                        .title(format!("{} (cancelled)", event.name)),
                )))
            }
            SessionTransportEvent::ToolFinished(event) => {
                let output = self
                    .tool_output
                    .remove(&event.call_id)
                    .or_else(|| event.output.clone());
                let status = match event.outcome {
                    ToolOutcome::Success => ToolCallStatus::Completed,
                    ToolOutcome::Failure => ToolCallStatus::Failed,
                };
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                    event.call_id.clone(),
                    ToolCallUpdateFields::new()
                        .status(status)
                        .title(event.summary.clone())
                        .content(output.map(|output| vec![text_content(output)])),
                )))
            }
            SessionTransportEvent::TokenUsage(event)
            | SessionTransportEvent::SessionTokenUsage(event)
                if event.context_window_tokens > 0 =>
            {
                Some(SessionUpdate::UsageUpdate(UsageUpdate::new(
                    event.used_tokens,
                    event.context_window_tokens,
                )))
            }
            SessionTransportEvent::SessionTitleUpdated { title, .. } => Some(
                SessionUpdate::SessionInfoUpdate(SessionInfoUpdate::new().title(title.clone())),
            ),
            SessionTransportEvent::TodoSnapshot(event) => {
                Some(SessionUpdate::Plan(plan(&event.items)))
            }
            _ => None,
        }
    }
}

/// Projects a todo snapshot onto an ACP plan.
///
/// ACP replaces the whole plan with every update, which is how a snapshot
/// reports the todo list, so each entry carries the status the snapshot gave it.
///
/// The two vocabularies do not line up, and the projection cannot recover what
/// it drops:
///
/// - `blocked` work is reported as pending. It is still open and is not being
///   worked on, and ACP has no state of its own for waiting on something else.
/// - `cancelled` items are reported as completed. They are no longer
///   outstanding, and ACP cannot tell an abandoned item from a finished one.
/// - Every entry is ranked medium. ACP requires a priority and the todo model
///   tracks none, so no entry is ordered against another.
/// - Todo ids are dropped; ACP entries are identified by their content alone.
fn plan(items: &[TodoItem]) -> Plan {
    Plan::new(items.iter().map(plan_entry).collect())
}

fn plan_entry(item: &TodoItem) -> PlanEntry {
    PlanEntry::new(
        item.content.clone(),
        PlanEntryPriority::Medium,
        plan_status(&item.status),
    )
}

fn plan_status(status: &TodoStatus) -> PlanEntryStatus {
    match status {
        TodoStatus::Pending | TodoStatus::Blocked => PlanEntryStatus::Pending,
        TodoStatus::InProgress => PlanEntryStatus::InProgress,
        TodoStatus::Completed | TodoStatus::Cancelled => PlanEntryStatus::Completed,
    }
}

fn user_message_text(event: &UserMessageEvent) -> String {
    event.content.display_text()
}

pub(super) fn message_chunk(text: String) -> ContentChunk {
    ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
}

fn text_content(text: String) -> ToolCallContent {
    ToolCallContent::Content(Content::new(ContentBlock::Text(TextContent::new(text))))
}

/// Maps a project tool name onto the closest ACP tool kind.
fn tool_kind(name: &str) -> ToolKind {
    match name {
        tool_names::TOOL_FS_READ
        | tool_names::TOOL_FS_LIST
        | tool_names::TOOL_SKILL_RESOURCE_LIST
        | tool_names::TOOL_SKILL_RESOURCE_READ
        | tool_names::TOOL_MEMORY_RECALL
        | tool_names::TOOL_GIT_STATUS
        | tool_names::TOOL_GIT_DIFF
        | tool_names::TOOL_GIT_LOG
        | tool_names::TOOL_AGENT_STATUS
        | tool_names::TOOL_AGENT_WAIT => ToolKind::Read,
        tool_names::TOOL_FS_WRITE
        | tool_names::TOOL_FS_APPEND
        | tool_names::TOOL_FS_MKDIR
        | tool_names::TOOL_EDIT_APPLY_PATCH
        | tool_names::TOOL_CODE_AST_REPLACE_PREVIEW => ToolKind::Edit,
        tool_names::TOOL_SEARCH_RG | tool_names::TOOL_CODE_AST_SEARCH => ToolKind::Search,
        tool_names::TOOL_SHELL_EXEC => ToolKind::Execute,
        tool_names::TOOL_WEB_FETCH => ToolKind::Fetch,
        tool_names::TOOL_WORKFLOW_TODOS
        | tool_names::TOOL_WORKFLOW_AUTO_CONTINUE
        | tool_names::TOOL_SKILL => ToolKind::Think,
        _ => ToolKind::Other,
    }
}

/// Renders tool arguments into the structured input ACP clients display.
/// Arguments that are not a JSON value are reported as unstructured text.
fn parse_arguments(arguments: Option<&str>) -> Option<serde_json::Value> {
    let arguments = arguments?;
    match serde_json::from_str(arguments) {
        Ok(value) => Some(value),
        Err(_) => Some(serde_json::Value::String(arguments.to_string())),
    }
}

#[cfg(test)]
mod tests {
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
        let update = projection.project(&SessionTransportEvent::ToolPending(
            ToolPendingEvent::new("call-1", tool_names::TOOL_SHELL_EXEC),
        ));
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
}
