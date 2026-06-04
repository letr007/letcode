use super::events::{
    AssistantDeltaEvent, ErrorEvent, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ReasoningDeltaEvent, ToolFinishedEvent, ToolOutcome,
    ToolStartedEvent, UserMessageEvent,
};
use crate::agent::{ConversationMessage, ConversationRole};
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptEvent, TranscriptRecord};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineItem {
    User(MessageView),
    Reasoning(ReasoningView),
    Assistant(MessageView),
    Tool(ToolView),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionStatus {
    Running,
    Succeeded,
    Failed,
}

impl ToolExecutionStatus {
    pub fn label(self) -> &'static str {
        match self {
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Timeline {
    items: Vec<TimelineItem>,
}

impl Timeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_conversation(messages: Vec<ConversationMessage>) -> Self {
        let mut timeline = Self::new();
        for message in messages {
            match message.role {
                ConversationRole::User => timeline.items.push(TimelineItem::User(MessageView {
                    id: None,
                    role: MessageRole::User,
                    text: message.content,
                    streaming: false,
                })),
                ConversationRole::Assistant => {
                    timeline.items.push(TimelineItem::Assistant(MessageView {
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
        for record in records {
            match &record.event {
                TranscriptEvent::UserMessage { content } => {
                    timeline.items.push(TimelineItem::User(MessageView {
                        id: None,
                        role: MessageRole::User,
                        text: content.clone(),
                        streaming: false,
                    }));
                }
                TranscriptEvent::AssistantMessage { content } => {
                    timeline.items.push(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: content.clone(),
                        streaming: false,
                    }));
                }
                TranscriptEvent::ReasoningMessage { content } => {
                    timeline.items.push(TimelineItem::Reasoning(ReasoningView {
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
                TranscriptEvent::PermissionDecision {
                    call_id,
                    tool,
                    args,
                    allowed,
                    reason,
                } => {
                    timeline
                        .items
                        .push(TimelineItem::Permission(PermissionView {
                            call_id: call_id.clone().unwrap_or_else(|| tool.clone()),
                            tool_name: tool.clone(),
                            summary: format_tool_call(tool, args),
                            arguments: Some(args.to_string()),
                            rationale: None,
                            status: if *allowed {
                                PermissionPromptStatus::Approved
                            } else {
                                PermissionPromptStatus::Denied
                            },
                            resolution_reason: reason.clone(),
                        }));
                }
                TranscriptEvent::Error { message } => {
                    timeline.items.push(TimelineItem::Error(ErrorView {
                        message: message.clone(),
                        details: None,
                    }));
                }
                TranscriptEvent::SessionStarted { .. }
                | TranscriptEvent::ModelChanged { .. }
                | TranscriptEvent::PermissionModeChanged { .. }
                | TranscriptEvent::Evidence { .. } => {}
            }
        }
        timeline
    }

    pub fn items(&self) -> &[TimelineItem] {
        &self.items
    }

    pub fn into_items(self) -> Vec<TimelineItem> {
        self.items
    }

    pub fn push_user_message(&mut self, event: UserMessageEvent) {
        self.items.push(TimelineItem::User(MessageView {
            id: event.message_id,
            role: MessageRole::User,
            text: event.content,
            streaming: false,
        }));
    }

    pub fn push_assistant_delta(&mut self, event: AssistantDeltaEvent) {
        if let Some(message) = self.active_assistant_message_mut(event.message_id.as_deref()) {
            message.text.push_str(&event.delta);
            return;
        }

        self.items.push(TimelineItem::Assistant(MessageView {
            id: event.message_id,
            role: MessageRole::Assistant,
            text: event.delta,
            streaming: true,
        }));
    }

    pub fn push_reasoning_delta(&mut self, event: ReasoningDeltaEvent) {
        if let Some(reasoning) = self.active_reasoning_mut(&event.item_id) {
            reasoning.text.push_str(&event.delta);
            return;
        }

        self.items.push(TimelineItem::Reasoning(ReasoningView {
            item_id: event.item_id,
            text: event.delta,
            streaming: true,
        }));
    }

    pub fn finalize_reasoning(&mut self, item_id: &str, text: &str) {
        if let Some(reasoning) = self.find_reasoning_mut(item_id) {
            reasoning.text = text.to_string();
            reasoning.streaming = false;
            return;
        }

        self.items.push(TimelineItem::Reasoning(ReasoningView {
            item_id: item_id.to_string(),
            text: text.to_string(),
            streaming: false,
        }));
    }

    pub fn finalize_assistant_message(&mut self, message_id: Option<&str>) {
        if let Some(message) = self.active_assistant_message_mut(message_id) {
            message.streaming = false;
        }
    }

    pub fn push_tool_started(&mut self, event: ToolStartedEvent) {
        self.items.push(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: event.summary,
            arguments: event.arguments,
            output: None,
            status: ToolExecutionStatus::Running,
        }));
    }

    pub fn push_tool_finished(&mut self, event: ToolFinishedEvent) {
        if let Some(tool) = self.find_tool_mut(&event.call_id) {
            tool.name = event.name;
            tool.summary = event.summary;
            tool.output = event.output;
            tool.status = match event.outcome {
                ToolOutcome::Success => ToolExecutionStatus::Succeeded,
                ToolOutcome::Failure => ToolExecutionStatus::Failed,
            };
            return;
        }

        self.items.push(TimelineItem::Tool(ToolView {
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
        self.items
            .push(TimelineItem::Permission(PermissionView::from_request(
                event,
            )));
    }

    pub fn resolve_permission(&mut self, event: PermissionResolutionEvent) {
        if let Some(permission) = self.find_permission_mut(&event.call_id) {
            permission.status = match event.decision {
                PermissionDecision::Approved => PermissionPromptStatus::Approved,
                PermissionDecision::Denied => PermissionPromptStatus::Denied,
            };
            permission.resolution_reason = event.reason;
            return;
        }

        self.items.push(TimelineItem::Permission(PermissionView {
            call_id: event.call_id,
            tool_name: "unknown tool".into(),
            summary: "Permission resolved without an earlier prompt in timeline".into(),
            arguments: None,
            rationale: None,
            status: match event.decision {
                PermissionDecision::Approved => PermissionPromptStatus::Approved,
                PermissionDecision::Denied => PermissionPromptStatus::Denied,
            },
            resolution_reason: event.reason,
        }));
    }

    pub fn push_error(&mut self, event: ErrorEvent) {
        self.items.push(TimelineItem::Error(ErrorView {
            message: event.message,
            details: event.details,
        }));
    }

    pub fn push_notice(&mut self, message: impl Into<String>) {
        self.items.push(TimelineItem::Notice(NoticeView {
            message: message.into(),
        }));
    }

    pub fn active_tool(&self) -> Option<&ToolView> {
        self.items.iter().rev().find_map(|item| match item {
            TimelineItem::Tool(tool) if tool.status == ToolExecutionStatus::Running => Some(tool),
            _ => None,
        })
    }

    fn active_assistant_message_mut(
        &mut self,
        message_id: Option<&str>,
    ) -> Option<&mut MessageView> {
        self.items.iter_mut().rev().find_map(|item| match item {
            TimelineItem::Assistant(message)
                if message.streaming
                    && (message_id.is_none() || message.id.as_deref() == message_id) =>
            {
                Some(message)
            }
            _ => None,
        })
    }

    fn find_tool_mut(&mut self, call_id: &str) -> Option<&mut ToolView> {
        self.items.iter_mut().find_map(|item| match item {
            TimelineItem::Tool(tool) if tool.call_id == call_id => Some(tool),
            _ => None,
        })
    }

    fn find_permission_mut(&mut self, call_id: &str) -> Option<&mut PermissionView> {
        self.items.iter_mut().find_map(|item| match item {
            TimelineItem::Permission(permission) if permission.call_id == call_id => {
                Some(permission)
            }
            _ => None,
        })
    }

    fn active_reasoning_mut(&mut self, item_id: &str) -> Option<&mut ReasoningView> {
        self.items.iter_mut().rev().find_map(|item| match item {
            TimelineItem::Reasoning(reasoning)
                if reasoning.streaming && reasoning.item_id == item_id =>
            {
                Some(reasoning)
            }
            _ => None,
        })
    }

    fn find_reasoning_mut(&mut self, item_id: &str) -> Option<&mut ReasoningView> {
        self.items.iter_mut().find_map(|item| match item {
            TimelineItem::Reasoning(reasoning) if reasoning.item_id == item_id => Some(reasoning),
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn permission_resolution_updates_existing_prompt() {
        let mut timeline = Timeline::new();

        timeline.push_permission_request(PermissionRequestEvent {
            call_id: "call-2".into(),
            tool_name: "git".into(),
            summary: "commit changes".into(),
            arguments: Some("git commit -m test".into()),
            rationale: Some("Needed to save work".into()),
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
}
