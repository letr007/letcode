use super::events::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, PermissionDecision,
    PermissionRequestEvent, PermissionResolutionEvent, ReasoningDeltaEvent, TodoSnapshotEvent,
    ToolFinishedEvent, ToolOutcome, ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
use crate::agent::{AutoContinueState, ConversationMessage, ConversationRole, TodoItem};
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineItem {
    User(MessageView),
    Reasoning(ReasoningView),
    Assistant(MessageView),
    Tool(ToolView),
    Todo(TodoView),
    Permission(PermissionView),
    Error(ErrorView),
    Notice(NoticeView),
}

impl TimelineItem {
    pub fn blocks(&self) -> Vec<DisplayBlock> {
        match self {
            Self::User(message) | Self::Assistant(message) => vec![DisplayBlock::Paragraph {
                role: message.role,
                text: message.text.clone(),
                streaming: message.streaming,
            }],
            Self::Reasoning(reasoning) => vec![DisplayBlock::StatusLine {
                label: if reasoning.streaming {
                    "thinking"
                } else {
                    "thought"
                }
                .into(),
                text: reasoning.text.clone(),
            }],
            Self::Tool(tool) => {
                let mut blocks = vec![DisplayBlock::StatusLine {
                    label: tool.status.label().to_string(),
                    text: format!("{} — {}", tool.name, tool.summary),
                }];

                if let Some(arguments) = &tool.arguments {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "args".into(),
                        value: arguments.clone(),
                    });
                }

                if let Some(output) = &tool.output {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "output".into(),
                        value: output.clone(),
                    });
                }

                blocks
            }
            Self::Todo(todo) => {
                let summary = if todo.items.is_empty() {
                    "0 items".into()
                } else {
                    format!(
                        "{} items · {} pending · {} in progress · {} done",
                        todo.items.len(),
                        todo.items
                            .iter()
                            .filter(|item| item.status == crate::agent::TodoStatus::Pending)
                            .count(),
                        todo.items
                            .iter()
                            .filter(|item| item.status == crate::agent::TodoStatus::InProgress)
                            .count(),
                        todo.items
                            .iter()
                            .filter(|item| item.status == crate::agent::TodoStatus::Completed)
                            .count(),
                    )
                };
                let mut blocks = vec![DisplayBlock::StatusLine {
                    label: "todo".into(),
                    text: summary,
                }];

                blocks.push(DisplayBlock::KeyValue {
                    label: "auto".into(),
                    value: if todo.auto_continue.enabled {
                        format!("on · max {}", todo.auto_continue.max_continuations)
                    } else {
                        "off".into()
                    },
                });

                for item in &todo.items {
                    blocks.push(DisplayBlock::KeyValue {
                        label: item.id.clone(),
                        value: format!("{:?} · {}", item.status, item.content),
                    });
                }

                blocks
            }
            Self::Permission(permission) => {
                let mut blocks = vec![DisplayBlock::StatusLine {
                    label: permission.status.label().to_string(),
                    text: format!("{} — {}", permission.tool_name, permission.summary),
                }];

                if let Some(arguments) = &permission.arguments {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "args".into(),
                        value: arguments.clone(),
                    });
                }

                if let Some(rationale) = &permission.rationale {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "why".into(),
                        value: rationale.clone(),
                    });
                }

                if let Some(reason) = &permission.resolution_reason {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "resolution".into(),
                        value: reason.clone(),
                    });
                }

                blocks
            }
            Self::Error(error) => {
                let mut blocks = vec![DisplayBlock::StatusLine {
                    label: "error".into(),
                    text: error.message.clone(),
                }];

                if let Some(details) = &error.details {
                    blocks.push(DisplayBlock::KeyValue {
                        label: "details".into(),
                        value: details.clone(),
                    });
                }

                blocks
            }
            Self::Notice(notice) => vec![DisplayBlock::StatusLine {
                label: "notice".into(),
                text: notice.message.clone(),
            }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayBlock {
    Paragraph {
        role: MessageRole,
        text: String,
        streaming: bool,
    },
    StatusLine {
        label: String,
        text: String,
    },
    KeyValue {
        label: String,
        value: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageView {
    pub id: Option<String>,
    pub role: MessageRole,
    pub text: String,
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningView {
    pub item_id: String,
    pub text: String,
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolView {
    pub call_id: String,
    pub name: String,
    pub summary: String,
    pub arguments: Option<String>,
    pub output: Option<String>,
    pub status: ToolExecutionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoView {
    pub items: Vec<TodoItem>,
    pub auto_continue: AutoContinueState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

impl ToolExecutionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "tool",
            Self::Running => "tool",
            Self::Succeeded => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionView {
    pub call_id: String,
    pub tool_name: String,
    pub summary: String,
    pub arguments: Option<String>,
    pub rationale: Option<String>,
    pub origin_label: Option<String>,
    pub status: PermissionPromptStatus,
    pub resolution_reason: Option<String>,
}

impl PermissionView {
    pub fn from_request(event: PermissionRequestEvent) -> Self {
        Self {
            call_id: event.call_id,
            tool_name: event.tool_name,
            summary: event.summary,
            arguments: event.arguments,
            rationale: event.rationale,
            origin_label: event.origin_label,
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionPromptStatus {
    Pending,
    Approved,
    Denied,
}

impl PermissionPromptStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "approval",
            Self::Approved => "approved",
            Self::Denied => "denied",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorView {
    pub message: String,
    pub details: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeView {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Timeline {
    items: Vec<TimelineItem>,
    revisions: Vec<u64>,
    next_revision: u64,
    cache_id: u64,
}

impl PartialEq for Timeline {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl Eq for Timeline {}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            revisions: Vec::new(),
            next_revision: 0,
            cache_id: next_timeline_cache_id(),
        }
    }
}

impl Timeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_conversation(messages: Vec<ConversationMessage>) -> Self {
        let mut timeline = Self::new();
        for message in messages {
            match message.role {
                ConversationRole::User => timeline.push_item(TimelineItem::User(MessageView {
                    id: None,
                    role: MessageRole::User,
                    text: message.content,
                    streaming: false,
                })),
                ConversationRole::Assistant => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: message.content,
                        streaming: false,
                    }))
                }
            }
        }
        timeline
    }

    pub fn from_transcript_records(records: &[TranscriptRecord]) -> Self {
        let mut timeline = Self::new();
        let mut current_auto_continue = AutoContinueState::default();
        for record in records {
            match &record.event {
                TranscriptEvent::UserMessage { content } => {
                    timeline.push_item(TimelineItem::User(MessageView {
                        id: None,
                        role: MessageRole::User,
                        text: content.clone(),
                        streaming: false,
                    }));
                }
                TranscriptEvent::AssistantMessage { content } => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: content.clone(),
                        streaming: false,
                    }));
                }
                TranscriptEvent::ReasoningMessage { content } => {
                    timeline.push_item(TimelineItem::Reasoning(ReasoningView {
                        item_id: format!("restored-reasoning-{}", record.sequence),
                        text: content.clone(),
                        streaming: false,
                    }));
                }
                TranscriptEvent::ToolCallStarted {
                    call_id,
                    name,
                    args,
                } => {
                    timeline.push_tool_started(ToolStartedEvent {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        summary: format_tool_call(name, args),
                        arguments: Some(args.to_string()),
                    });
                }
                TranscriptEvent::ToolCallFinished {
                    call_id,
                    name,
                    ok,
                    output,
                } => {
                    timeline.push_tool_finished(ToolFinishedEvent {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        summary: restored_tool_summary(name, *ok),
                        outcome: if *ok {
                            ToolOutcome::Success
                        } else {
                            ToolOutcome::Failure
                        },
                        output: serde_json::to_value(output)
                            .ok()
                            .map(|value| value.to_string()),
                    });
                }
                TranscriptEvent::TodoSnapshot { items } => {
                    timeline.push_todo_snapshot(TodoSnapshotEvent::new(items.clone()));
                    timeline.apply_auto_continue_changed(AutoContinueChangedEvent::new(
                        current_auto_continue.clone(),
                    ));
                }
                TranscriptEvent::AutoContinueChanged { state } => {
                    current_auto_continue = state.clone();
                    timeline
                        .apply_auto_continue_changed(AutoContinueChangedEvent::new(state.clone()));
                }
                TranscriptEvent::PermissionDecision {
                    call_id,
                    tool,
                    args,
                    allowed,
                    reason,
                } => {
                    timeline.push_item(TimelineItem::Permission(PermissionView {
                        call_id: call_id.clone().unwrap_or_else(|| tool.clone()),
                        tool_name: tool.clone(),
                        summary: format_tool_call(tool, args),
                        arguments: Some(args.to_string()),
                        rationale: None,
                        origin_label: None,
                        status: if *allowed {
                            PermissionPromptStatus::Approved
                        } else {
                            PermissionPromptStatus::Denied
                        },
                        resolution_reason: reason.clone(),
                    }));
                }
                TranscriptEvent::Error { message } => {
                    timeline.push_item(TimelineItem::Error(ErrorView {
                        message: message.clone(),
                        details: None,
                    }));
                }
                TranscriptEvent::SubagentResult { .. } => {}
                TranscriptEvent::SubagentLifecycle { .. } => {}
                TranscriptEvent::SessionStarted { .. }
                | TranscriptEvent::SessionTitle { .. }
                | TranscriptEvent::TurnStarted(_)
                | TranscriptEvent::ModelChanged { .. }
                | TranscriptEvent::PermissionModeChanged { .. }
                | TranscriptEvent::AutoContinuationScheduled { .. }
                | TranscriptEvent::ValidationAdvisory(_)
                | TranscriptEvent::ToolExecutionSummary(_)
                | TranscriptEvent::TurnFinalized(_)
                | TranscriptEvent::Evidence { .. }
                | TranscriptEvent::Unknown => {}
            }
        }
        timeline
    }

    pub fn items(&self) -> &[TimelineItem] {
        &self.items
    }

    pub fn item_revisions(&self) -> &[u64] {
        &self.revisions
    }

    pub fn cache_id(&self) -> u64 {
        self.cache_id
    }

    fn push_item(&mut self, item: TimelineItem) {
        self.items.push(item);
        self.revisions.push(self.next_revision);
        self.next_revision = self.next_revision.wrapping_add(1).max(1);
    }

    fn bump_revision(&mut self, index: usize) {
        if let Some(revision) = self.revisions.get_mut(index) {
            *revision = self.next_revision;
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
        }
    }

    pub fn push_user_message(&mut self, event: UserMessageEvent) {
        self.push_item(TimelineItem::User(MessageView {
            id: event.message_id,
            role: MessageRole::User,
            text: event.content,
            streaming: false,
        }));
    }

    pub fn push_assistant_delta(&mut self, event: AssistantDeltaEvent) {
        if let Some(index) = self.active_assistant_message_index(event.message_id.as_deref()) {
            if let TimelineItem::Assistant(message) = &mut self.items[index] {
                message.text.push_str(&event.delta);
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Assistant(MessageView {
            id: event.message_id,
            role: MessageRole::Assistant,
            text: event.delta,
            streaming: true,
        }));
    }

    pub fn push_reasoning_delta(&mut self, event: ReasoningDeltaEvent) {
        if let Some(index) = self.active_reasoning_index(&event.item_id) {
            if let TimelineItem::Reasoning(reasoning) = &mut self.items[index] {
                reasoning.text.push_str(&event.delta);
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id: event.item_id,
            text: event.delta,
            streaming: true,
        }));
    }

    pub fn finalize_reasoning(&mut self, item_id: &str, text: &str) {
        if let Some(index) = self.find_reasoning_index(item_id) {
            if let TimelineItem::Reasoning(reasoning) = &mut self.items[index] {
                reasoning.text = text.to_string();
                reasoning.streaming = false;
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id: item_id.to_string(),
            text: text.to_string(),
            streaming: false,
        }));
    }

    pub fn finalize_assistant_message(&mut self, message_id: Option<&str>) {
        if let Some(index) = self.active_assistant_message_index(message_id) {
            if let TimelineItem::Assistant(message) = &mut self.items[index] {
                message.streaming = false;
            }
            self.bump_revision(index);
        }
    }

    pub fn push_tool_started(&mut self, event: ToolStartedEvent) {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                tool.name = event.name;
                tool.summary = event.summary;
                tool.arguments = event.arguments;
                tool.output = None;
                tool.status = ToolExecutionStatus::Running;
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: event.summary,
            arguments: event.arguments,
            output: None,
            status: ToolExecutionStatus::Running,
        }));
    }

    pub fn push_tool_pending(&mut self, event: ToolPendingEvent) {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                if tool.status == ToolExecutionStatus::Pending {
                    tool.name = event.name;
                } else {
                    return;
                }
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: "preparing input".into(),
            arguments: None,
            output: None,
            status: ToolExecutionStatus::Pending,
        }));
    }

    pub fn push_tool_finished(&mut self, event: ToolFinishedEvent) {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                tool.name = event.name;
                tool.summary = event.summary;
                tool.output = event.output;
                tool.status = match event.outcome {
                    ToolOutcome::Success => ToolExecutionStatus::Succeeded,
                    ToolOutcome::Failure => ToolExecutionStatus::Failed,
                };
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: event.summary,
            arguments: None,
            output: event.output,
            status: match event.outcome {
                ToolOutcome::Success => ToolExecutionStatus::Succeeded,
                ToolOutcome::Failure => ToolExecutionStatus::Failed,
            },
        }));
    }

    pub fn push_permission_request(&mut self, event: PermissionRequestEvent) {
        self.push_item(TimelineItem::Permission(PermissionView::from_request(
            event,
        )));
    }

    pub fn push_todo_snapshot(&mut self, event: TodoSnapshotEvent) {
        self.push_item(TimelineItem::Todo(TodoView {
            items: event.items,
            auto_continue: AutoContinueState::default(),
        }));
    }

    pub fn apply_auto_continue_changed(&mut self, event: AutoContinueChangedEvent) {
        if let Some(index) = self.todo_view_index() {
            if let TimelineItem::Todo(todo) = &mut self.items[index] {
                todo.auto_continue = event.state;
            }
            self.bump_revision(index);
        }
    }

    pub fn resolve_permission(&mut self, event: PermissionResolutionEvent) {
        if let Some(index) = self.find_permission_index(&event.call_id) {
            if let TimelineItem::Permission(permission) = &mut self.items[index] {
                permission.status = match event.decision {
                    PermissionDecision::Approved => PermissionPromptStatus::Approved,
                    PermissionDecision::Denied => PermissionPromptStatus::Denied,
                };
                permission.resolution_reason = event.reason;
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Permission(PermissionView {
            call_id: event.call_id,
            tool_name: "unknown tool".into(),
            summary: "Permission resolved without an earlier prompt in timeline".into(),
            arguments: None,
            rationale: None,
            origin_label: None,
            status: match event.decision {
                PermissionDecision::Approved => PermissionPromptStatus::Approved,
                PermissionDecision::Denied => PermissionPromptStatus::Denied,
            },
            resolution_reason: event.reason,
        }));
    }

    pub fn push_error(&mut self, event: ErrorEvent) {
        self.push_item(TimelineItem::Error(ErrorView {
            message: event.message,
            details: event.details,
        }));
    }

    pub fn push_notice(&mut self, message: impl Into<String>) {
        self.push_item(TimelineItem::Notice(NoticeView {
            message: message.into(),
        }));
    }

    pub fn active_tool(&self) -> Option<&ToolView> {
        self.items.iter().rev().find_map(|item| match item {
            TimelineItem::Tool(tool)
                if matches!(
                    tool.status,
                    ToolExecutionStatus::Pending | ToolExecutionStatus::Running
                ) =>
            {
                Some(tool)
            }
            _ => None,
        })
    }

    fn active_assistant_message_index(&self, message_id: Option<&str>) -> Option<usize> {
        self.items.iter().rposition(|item| match item {
            TimelineItem::Assistant(message) => {
                message.streaming && (message_id.is_none() || message.id.as_deref() == message_id)
            }
            _ => false,
        })
    }

    fn find_tool_index(&self, call_id: &str) -> Option<usize> {
        self.items.iter().position(|item| match item {
            TimelineItem::Tool(tool) => tool.call_id == call_id,
            _ => false,
        })
    }

    fn find_permission_index(&self, call_id: &str) -> Option<usize> {
        self.items.iter().position(|item| match item {
            TimelineItem::Permission(permission) => permission.call_id == call_id,
            _ => false,
        })
    }

    fn active_reasoning_index(&self, item_id: &str) -> Option<usize> {
        self.items.iter().rposition(|item| match item {
            TimelineItem::Reasoning(reasoning) => {
                reasoning.streaming && reasoning.item_id == item_id
            }
            _ => false,
        })
    }

    fn find_reasoning_index(&self, item_id: &str) -> Option<usize> {
        self.items.iter().position(|item| match item {
            TimelineItem::Reasoning(reasoning) => reasoning.item_id == item_id,
            _ => false,
        })
    }

    fn todo_view_index(&self) -> Option<usize> {
        let current_turn_start = self
            .items
            .iter()
            .rposition(|item| matches!(item, TimelineItem::User(_)))
            .unwrap_or(0);

        self.items[current_turn_start..]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(offset, item)| match item {
                TimelineItem::Todo(_) => Some(current_turn_start + offset),
                _ => None,
            })
    }
}

fn restored_tool_summary(name: &str, ok: bool) -> String {
    if ok {
        format!("{name} completed")
    } else {
        format!("{name} failed")
    }
}

fn next_timeline_cache_id() -> u64 {
    static NEXT_TIMELINE_CACHE_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_TIMELINE_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::tool::ToolResult;
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use serde_json::json;

    #[test]
    fn assistant_deltas_merge_into_active_message_and_finalize() {
        let mut timeline = Timeline::new();

        timeline.push_assistant_delta(AssistantDeltaEvent {
            message_id: Some("msg-1".into()),
            delta: "Hello".into(),
        });
        timeline.push_assistant_delta(AssistantDeltaEvent {
            message_id: Some("msg-1".into()),
            delta: ", world".into(),
        });
        timeline.finalize_assistant_message(Some("msg-1"));

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Assistant(message) => {
                assert_eq!(message.text, "Hello, world");
                assert!(!message.streaming);
            }
            other => panic!("expected assistant item, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_deltas_merge_and_finalize() {
        let mut timeline = Timeline::new();

        timeline.push_reasoning_delta(ReasoningDeltaEvent::new("r-1", "Inspecting"));
        timeline.push_reasoning_delta(ReasoningDeltaEvent::new("r-1", " workflow"));
        timeline.finalize_reasoning("r-1", "Inspecting workflow");

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Reasoning(reasoning) => {
                assert_eq!(reasoning.text, "Inspecting workflow");
                assert!(!reasoning.streaming);
            }
            other => panic!("expected reasoning item, got {other:?}"),
        }
    }

    #[test]
    fn tool_finish_updates_existing_tool_status_and_output() {
        let mut timeline = Timeline::new();

        timeline.push_tool_started(ToolStartedEvent {
            call_id: "tool-1".into(),
            name: "shell__exec".into(),
            summary: "run ls".into(),
            arguments: Some("ls".into()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "tool-1".into(),
            name: "shell__exec".into(),
            summary: "run ls".into(),
            outcome: ToolOutcome::Success,
            output: Some("file-a\nfile-b".into()),
        });

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Tool(tool) => {
                assert_eq!(tool.status, ToolExecutionStatus::Succeeded);
                assert_eq!(tool.output.as_deref(), Some("file-a\nfile-b"));
                assert_eq!(tool.arguments.as_deref(), Some("ls"));
            }
            other => panic!("expected tool item, got {other:?}"),
        }
    }

    #[test]
    fn tool_pending_then_started_updates_same_timeline_item() {
        let mut timeline = Timeline::new();

        timeline.push_tool_pending(ToolPendingEvent::new("tool-1", "edit__apply_patch"));
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "tool-1".into(),
            name: "edit__apply_patch".into(),
            summary: "Apply patch".into(),
            arguments: Some(r#"{"patch":"..."}"#.into()),
        });

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Tool(tool) => {
                assert_eq!(tool.call_id, "tool-1");
                assert_eq!(tool.status, ToolExecutionStatus::Running);
                assert_eq!(tool.summary, "Apply patch");
                assert_eq!(tool.arguments.as_deref(), Some(r#"{"patch":"..."}"#));
            }
            other => panic!("expected tool item, got {other:?}"),
        }
    }

    #[test]
    fn late_duplicate_pending_does_not_regress_running_item() {
        let mut timeline = Timeline::new();

        timeline.push_tool_pending(ToolPendingEvent::new("tool-1", "workflow__todos"));
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "tool-1".into(),
            name: "workflow__todos".into(),
            summary: "update todo list".into(),
            arguments: Some(r#"{"items":[]}"#.into()),
        });
        timeline.push_tool_pending(ToolPendingEvent::new("tool-1", "workflow__todos"));

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Tool(tool) => {
                assert_eq!(tool.status, ToolExecutionStatus::Running);
                assert_eq!(tool.summary, "update todo list");
                assert_eq!(tool.arguments.as_deref(), Some(r#"{"items":[]}"#));
            }
            other => panic!("expected tool item, got {other:?}"),
        }
    }

    #[test]
    fn permission_resolution_updates_existing_prompt() {
        let mut timeline = Timeline::new();

        timeline.push_permission_request(PermissionRequestEvent {
            call_id: "call-2".into(),
            tool_name: "git".into(),
            summary: "commit changes".into(),
            arguments: Some("git commit -m test".into()),
            rationale: Some("Needed to save work".into()),
            origin_label: None,
        });
        timeline.resolve_permission(PermissionResolutionEvent::denied(
            "call-2",
            Some("not approved".into()),
        ));

        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Permission(permission) => {
                assert_eq!(permission.status, PermissionPromptStatus::Denied);
                assert_eq!(
                    permission.resolution_reason.as_deref(),
                    Some("not approved")
                );
            }
            other => panic!("expected permission item, got {other:?}"),
        }
    }

    #[test]
    fn transcript_restore_preserves_tool_arguments_and_output_for_visible_cards() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::ToolCallStarted {
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    args: json!({
                        "path": "tool-write-test.txt",
                        "content": "hello\n"
                    }),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::ToolCallFinished {
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    ok: true,
                    output: ToolResult::ok(
                        "fs__write",
                        json!({"path":"tool-write-test.txt","bytes_written":6}),
                    ),
                },
            },
        ];

        let timeline = Timeline::from_transcript_records(&records);
        let items = timeline.items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            TimelineItem::Tool(tool) => {
                assert_eq!(tool.status, ToolExecutionStatus::Succeeded);
                assert_eq!(tool.name, "fs__write");
                assert!(
                    tool.arguments
                        .as_deref()
                        .is_some_and(|args| args.contains("tool-write-test.txt"))
                );
                assert!(
                    tool.output
                        .as_deref()
                        .is_some_and(|output| output.contains("bytes_written"))
                );
            }
            other => panic!("expected restored tool item, got {other:?}"),
        }
    }

    #[test]
    fn todo_snapshot_appends_new_timeline_item() {
        let mut timeline = Timeline::new();

        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: TodoStatus::Pending,
        }]));
        timeline.apply_auto_continue_changed(AutoContinueChangedEvent::new(AutoContinueState {
            enabled: true,
            max_continuations: 2,
        }));
        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: TodoStatus::Completed,
        }]));
        timeline.apply_auto_continue_changed(AutoContinueChangedEvent::new(AutoContinueState {
            enabled: true,
            max_continuations: 2,
        }));

        assert_eq!(timeline.items().len(), 2);
        match &timeline.items()[0] {
            TimelineItem::Todo(todo) => {
                assert_eq!(todo.items[0].status, TodoStatus::Pending);
                assert!(todo.auto_continue.enabled);
            }
            other => panic!("expected todo item, got {other:?}"),
        }
        match &timeline.items()[1] {
            TimelineItem::Todo(todo) => {
                assert_eq!(todo.items[0].status, TodoStatus::Completed);
                assert!(todo.auto_continue.enabled);
            }
            other => panic!("expected todo item, got {other:?}"),
        }
    }

    #[test]
    fn todo_snapshot_creates_new_card_after_new_user_turn() {
        let mut timeline = Timeline::new();

        timeline.push_user_message(UserMessageEvent::new("first"));
        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "first todo".into(),
            status: TodoStatus::Completed,
        }]));
        timeline.push_user_message(UserMessageEvent::new("second"));
        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t2".into(),
            content: "second todo".into(),
            status: TodoStatus::Pending,
        }]));

        let todo_count = timeline
            .items()
            .iter()
            .filter(|item| matches!(item, TimelineItem::Todo(_)))
            .count();
        assert_eq!(todo_count, 2);
    }

    #[test]
    fn transcript_restore_keeps_latest_todo_and_auto_continue_state() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::AutoContinueChanged {
                    state: AutoContinueState {
                        enabled: true,
                        max_continuations: 3,
                    },
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::TodoSnapshot {
                    items: vec![TodoItem {
                        id: "t1".into(),
                        content: "inspect".into(),
                        status: TodoStatus::Pending,
                    }],
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                event: TranscriptEvent::TodoSnapshot {
                    items: vec![TodoItem {
                        id: "t1".into(),
                        content: "inspect".into(),
                        status: TodoStatus::Completed,
                    }],
                },
            },
        ];

        let timeline = Timeline::from_transcript_records(&records);
        assert_eq!(timeline.items().len(), 2);
        match &timeline.items()[1] {
            TimelineItem::Todo(todo) => {
                assert_eq!(todo.items[0].status, TodoStatus::Completed);
                assert!(todo.auto_continue.enabled);
                assert_eq!(todo.auto_continue.max_continuations, 3);
            }
            other => panic!("expected restored todo item, got {other:?}"),
        }
    }

    #[test]
    fn transcript_restore_renders_subagent_result_as_compact_tool_card_model() {
        let records = vec![TranscriptRecord {
            session_id: "parent-session-abcdef".into(),
            sequence: 1,
            timestamp_ms: 0,
            event: TranscriptEvent::SubagentResult {
                run_id: "run-1".into(),
                parent_session_id: "parent-session-abcdef".into(),
                parent_run_id: "turn-1".into(),
                child_session_id: "child-session-1234567890".into(),
                agent_name: "Explorer".into(),
                status: "completed".into(),
                summary: "scanned src/tool.rs".into(),
            },
        }];

        let timeline = Timeline::from_transcript_records(&records);
        assert_eq!(timeline.items().len(), 0);
    }

    #[test]
    fn transcript_restore_ignores_turn_audit_events() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::UserMessage {
                    content: "hello".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::TurnStarted(crate::agent::TurnStartedEvent {
                    turn_id: 1,
                    intent: "engineering".into(),
                    directive: "none".into(),
                    validation_reminder: "focused".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                event: TranscriptEvent::ToolExecutionSummary(
                    crate::agent::ToolExecutionSummaryEvent {
                        turn_id: 1,
                        call_id: "call-1".into(),
                        name: "fs__read".into(),
                        status: "executed".into(),
                        rejection: None,
                        effect_kind: "read".into(),
                        primary_path: Some("src/main.rs".into()),
                        command: None,
                    },
                ),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 3,
                event: TranscriptEvent::TurnFinalized(crate::agent::TurnFinalizedEvent {
                    turn_id: 1,
                    outcome: "completed".into(),
                    tool_call_count: 1,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 5,
                timestamp_ms: 4,
                event: TranscriptEvent::AssistantMessage {
                    content: "done".into(),
                },
            },
        ];

        let timeline = Timeline::from_transcript_records(&records);
        assert_eq!(timeline.items().len(), 2);
        assert!(matches!(timeline.items()[0], TimelineItem::User(_)));
        assert!(matches!(timeline.items()[1], TimelineItem::Assistant(_)));
    }
}
