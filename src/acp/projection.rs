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
        | tool_names::TOOL_GIT
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
#[path = "../../tests/internal/acp/projection/tests.rs"]
mod tests;
