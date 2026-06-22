use super::events::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, PermissionDecision,
    PermissionRequestEvent, PermissionResolutionEvent, ReasoningDeltaEvent, TodoSnapshotEvent,
    ToolFinishedEvent, ToolOutcome, ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
use crate::agent::{AutoContinueState, ConversationMessage, ConversationRole, TodoItem};
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptEvent, TranscriptRecord};

pub(crate) const COMPACTION_SEPARATOR_LABEL: &str = "Earlier messages compacted";
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineItem {
    User(MessageView),
    Delegation(DelegationView),
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
            Self::Delegation(delegation) => vec![DisplayBlock::StatusLine {
                label: "delegate".into(),
                text: format!("@{} — {}", delegation.agent_name, delegation.task),
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
    pub queued: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningView {
    pub item_id: String,
    pub text: String,
    pub streaming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegationView {
    pub agent_name: String,
    pub task: String,
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
    Cancelled,
    Succeeded,
    Failed,
}

impl ToolExecutionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "tool",
            Self::Running => "tool",
            Self::Cancelled => "cancelled",
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
                    queued: false,
                })),
                ConversationRole::Summary => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: message.content,
                        streaming: false,
                        queued: false,
                    }))
                }
                ConversationRole::Assistant => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: message.content,
                        streaming: false,
                        queued: false,
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
                        queued: false,
                    }));
                }
                TranscriptEvent::AssistantMessage { content } => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: content.clone(),
                        streaming: false,
                        queued: false,
                    }));
                }
                TranscriptEvent::ContextCompaction(event) => {
                    timeline.push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        role: MessageRole::Assistant,
                        text: event.summary.clone(),
                        streaming: false,
                        queued: false,
                    }));
                    timeline.push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
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
                TranscriptEvent::ToolCallCancelled { call_id, name } => {
                    timeline.cancel_tool(call_id, name);
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
                TranscriptEvent::TurnInterrupted { .. } => {
                    timeline.cancel_active_tools();
                    timeline.push_notice("Interrupted by user");
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
                | TranscriptEvent::Evidence { .. }
                | TranscriptEvent::Unknown => {}
                TranscriptEvent::TurnFinalized(event) => {
                    if event.outcome == "interrupted" {
                        timeline.cancel_active_tools();
                        timeline.push_notice("Interrupted by user");
                    }
                }
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
        let revision = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1).max(1);

        if is_queued_user_item(&item) {
            self.items.push(item);
            self.revisions.push(revision);
            return;
        }

        if let Some(index) = self.items.iter().position(is_queued_user_item) {
            self.items.insert(index, item);
            self.revisions.insert(index, revision);
            return;
        }

        self.items.push(item);
        self.revisions.push(revision);
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
            queued: event.queued,
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
            queued: false,
        }));
    }

    pub fn activate_first_queued_user_message(&mut self, content: &str) -> bool {
        let Some(index) = self.items.iter().position(|item| {
            matches!(
                item,
                TimelineItem::User(MessageView {
                    text,
                    queued: true,
                    ..
                }) if text == content
            )
        }) else {
            return false;
        };

        if let TimelineItem::User(message) = &mut self.items[index] {
            message.queued = false;
        }
        self.bump_revision(index);
        true
    }

    pub fn activate_queued_user_message_previews(&mut self) -> usize {
        let mut activated = 0usize;
        for index in 0..self.items.len() {
            if let TimelineItem::User(message) = &mut self.items[index]
                && message.queued
            {
                message.queued = false;
                self.bump_revision(index);
                activated = activated.saturating_add(1);
            }
        }
        activated
    }

    pub fn remove_queued_user_message_previews(&mut self) {
        let original_len = self.items.len();
        let mut retained_items = Vec::with_capacity(original_len);
        let mut retained_revisions = Vec::with_capacity(self.revisions.len());

        for (item, revision) in self.items.drain(..).zip(self.revisions.drain(..)) {
            let remove = matches!(&item, TimelineItem::User(MessageView { queued: true, .. }));
            if !remove {
                retained_items.push(item);
                retained_revisions.push(revision);
            }
        }

        self.items = retained_items;
        self.revisions = retained_revisions;
    }

    pub fn remove_first_queued_user_message_preview(&mut self, content: &str) -> bool {
        let Some(index) = self.items.iter().position(|item| {
            matches!(
                item,
                TimelineItem::User(MessageView {
                    text,
                    queued: true,
                    ..
                }) if text == content
            )
        }) else {
            return false;
        };

        self.items.remove(index);
        self.revisions.remove(index);
        true
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

    pub fn push_delegation(&mut self, agent_name: impl Into<String>, task: impl Into<String>) {
        self.push_item(TimelineItem::Delegation(DelegationView {
            agent_name: agent_name.into(),
            task: task.into(),
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

    pub fn push_tool_started(&mut self, event: ToolStartedEvent) -> bool {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                if tool.status == ToolExecutionStatus::Cancelled {
                    return false;
                }
                tool.name = event.name;
                tool.summary = event.summary;
                tool.arguments = event.arguments;
                tool.output = None;
                tool.status = ToolExecutionStatus::Running;
            }
            self.bump_revision(index);
            return true;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: event.summary,
            arguments: event.arguments,
            output: None,
            status: ToolExecutionStatus::Running,
        }));
        true
    }

    pub fn push_tool_pending(&mut self, event: ToolPendingEvent) -> bool {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                if tool.status == ToolExecutionStatus::Cancelled {
                    return false;
                }
                if tool.status == ToolExecutionStatus::Pending {
                    tool.name = event.name;
                } else {
                    return false;
                }
            }
            self.bump_revision(index);
            return true;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: event.call_id,
            name: event.name,
            summary: "preparing input".into(),
            arguments: None,
            output: None,
            status: ToolExecutionStatus::Pending,
        }));
        true
    }

    pub fn push_tool_finished(&mut self, event: ToolFinishedEvent) -> bool {
        if let Some(index) = self.find_tool_index(&event.call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index] {
                if tool.status == ToolExecutionStatus::Cancelled {
                    return false;
                }
                tool.name = event.name;
                tool.summary = event.summary;
                tool.output = event.output;
                tool.status = match event.outcome {
                    ToolOutcome::Success => ToolExecutionStatus::Succeeded,
                    ToolOutcome::Failure => ToolExecutionStatus::Failed,
                };
            }
            self.bump_revision(index);
            return true;
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
        true
    }

    pub fn cancel_active_tools(&mut self) -> usize {
        let mut cancelled = 0usize;
        for index in 0..self.items.len() {
            if let TimelineItem::Tool(tool) = &mut self.items[index]
                && matches!(
                    tool.status,
                    ToolExecutionStatus::Pending | ToolExecutionStatus::Running
                )
            {
                tool.status = ToolExecutionStatus::Cancelled;
                if tool.summary.is_empty() {
                    tool.summary = format!("{} cancelled", tool.name);
                }
                self.bump_revision(index);
                cancelled = cancelled.saturating_add(1);
            }
        }
        cancelled
    }

    pub fn cancel_tool(&mut self, call_id: &str, name: &str) {
        if let Some(index) = self.find_tool_index(call_id) {
            if let TimelineItem::Tool(tool) = &mut self.items[index]
                && matches!(
                    tool.status,
                    ToolExecutionStatus::Pending | ToolExecutionStatus::Running
                )
            {
                tool.name = name.to_string();
                tool.summary = format!("{name} cancelled");
                tool.status = ToolExecutionStatus::Cancelled;
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Tool(ToolView {
            call_id: call_id.to_string(),
            name: name.to_string(),
            summary: format!("{name} cancelled"),
            arguments: None,
            output: None,
            status: ToolExecutionStatus::Cancelled,
        }));
    }

    pub fn active_tool_calls(&self) -> Vec<(String, String)> {
        self.items
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool)
                    if matches!(
                        tool.status,
                        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
                    ) =>
                {
                    Some((tool.call_id.clone(), tool.name.clone()))
                }
                _ => None,
            })
            .collect()
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

    pub fn push_compaction_separator(&mut self, label: impl AsRef<str>) {
        self.push_notice(compaction_separator(label.as_ref()));
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
            .rposition(|item| matches!(item, TimelineItem::User(message) if !message.queued))
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

fn is_queued_user_item(item: &TimelineItem) -> bool {
    matches!(item, TimelineItem::User(message) if message.queued)
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

pub(crate) fn compaction_separator(label: &str) -> String {
    format!("──────── {label} ────────")
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
    fn transcript_restore_wraps_context_compaction_with_separators() {
        let records = vec![TranscriptRecord {
            session_id: "session".into(),
            sequence: 1,
            timestamp_ms: 0,
            event: TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                summary: "目标\n- 继续任务".into(),
                tail_start_index: 2,
                original_history_items: 8,
                retained_history_items: 3,
            }),
        }];

        let timeline = Timeline::from_transcript_records(&records);
        let items = timeline.items();

        assert_eq!(items.len(), 3);
        assert!(matches!(
            &items[0],
            TimelineItem::Notice(notice) if notice.message.contains(COMPACTION_SEPARATOR_LABEL)
        ));
        assert!(matches!(
            &items[1],
            TimelineItem::Assistant(message) if message.text == "目标\n- 继续任务"
        ));
        assert!(matches!(
            &items[2],
            TimelineItem::Notice(notice) if notice.message.contains(COMPACTION_SEPARATOR_LABEL)
        ));
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
    fn queued_user_previews_stay_after_later_turn_items() {
        let mut timeline = Timeline::new();

        timeline.push_user_message(UserMessageEvent::new("current"));
        timeline.push_assistant_delta(AssistantDeltaEvent {
            message_id: Some("assistant-1".into()),
            delta: "working".into(),
        });
        timeline.push_user_message(UserMessageEvent::queued("follow up"));
        timeline.push_notice("tool result visible above queued prompt");

        let items = timeline.items();
        assert!(
            matches!(items.last(), Some(TimelineItem::User(message)) if message.text == "follow up" && message.queued)
        );
        assert!(
            matches!(items.get(items.len() - 2), Some(TimelineItem::Notice(notice)) if notice.message == "tool result visible above queued prompt")
        );
    }

    #[test]
    fn queued_user_previews_do_not_split_current_todo_updates() {
        let mut timeline = Timeline::new();

        timeline.push_user_message(UserMessageEvent::new("current"));
        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "todo-1".into(),
            content: "work".into(),
            status: TodoStatus::InProgress,
        }]));
        timeline.push_user_message(UserMessageEvent::queued("follow up"));
        timeline.apply_auto_continue_changed(AutoContinueChangedEvent::new(AutoContinueState {
            enabled: true,
            max_continuations: 3,
        }));

        let todo = timeline
            .items()
            .iter()
            .find_map(|item| match item {
                TimelineItem::Todo(todo) => Some(todo),
                _ => None,
            })
            .expect("todo item remains visible");
        assert!(todo.auto_continue.enabled);
        assert!(matches!(
            timeline.items().last(),
            Some(TimelineItem::User(message)) if message.queued
        ));
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

    #[test]
    fn interrupt_cancels_only_active_tools() {
        let mut timeline = Timeline::new();
        timeline.push_tool_pending(ToolPendingEvent::new("call-pending", "shell__exec"));
        timeline.push_tool_started(ToolStartedEvent::new(
            "call-running",
            "fs__write",
            "write file",
        ));
        timeline.push_tool_finished(ToolFinishedEvent::new(
            "call-done",
            "fs__read",
            "read file",
            ToolOutcome::Success,
        ));

        assert_eq!(timeline.cancel_active_tools(), 2);

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools[0].status, ToolExecutionStatus::Cancelled);
        assert_eq!(tools[1].status, ToolExecutionStatus::Cancelled);
        assert_eq!(tools[2].status, ToolExecutionStatus::Succeeded);
    }

    #[test]
    fn transcript_restore_closes_interrupted_tool_turns() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "sleep 10"}),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
            },
        ];

        let timeline = Timeline::from_transcript_records(&records);
        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
        assert!(matches!(
            timeline.items().last(),
            Some(TimelineItem::Notice(notice)) if notice.message == "Interrupted by user"
        ));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn cancelled_tool_ignores_late_finished_event() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "call-1",
            "shell__exec",
            "run command",
        ));
        timeline.cancel_active_tools();

        timeline.push_tool_finished(ToolFinishedEvent::new(
            "call-1",
            "shell__exec",
            "shell__exec completed",
            ToolOutcome::Success,
        ));

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
    }
}
