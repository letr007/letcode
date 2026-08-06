use super::events::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, PermissionDecision,
    PermissionRequestEvent, PermissionResolutionEvent, ReasoningDeltaEvent, TodoSnapshotEvent,
    ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent,
    UserMessageEvent,
};
use crate::agent::{
    AutoContinueState, TodoItem, agent_name_for_subagent_tool, is_subagent_tool_name,
};
#[cfg(test)]
use crate::agent::{ConversationMessage, ConversationRole};
use crate::transcript::TranscriptRecord;
use crate::user_content::UserImageAttachment;

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
    /// Durable compaction block: drawn rules + markdown summary (streaming or final).
    Compaction(CompactionView),
}

#[cfg(test)]
impl TimelineItem {
    pub fn blocks(&self) -> Vec<DisplayBlock> {
        match self {
            Self::User(message) | Self::Assistant(message) => vec![DisplayBlock::Paragraph {
                role: message.role,
                text: message.display_text(),
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
                        "on".into()
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
            Self::Compaction(_) => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextOpenDetailView {
    pub title: String,
    pub badges: Vec<String>,
    pub lines: Vec<String>,
}

#[cfg(test)]
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
    pub submission_id: Option<String>,
    pub role: MessageRole,
    pub text: String,
    pub attachments: Vec<UserImageAttachment>,
    pub selected_skills: Vec<String>,
    pub streaming: bool,
    pub queued: bool,
}

#[cfg(test)]
impl MessageView {
    pub fn display_text(&self) -> String {
        if self.attachments.is_empty() && self.selected_skills.is_empty() {
            return self.text.clone();
        }

        let mut lines = Vec::new();
        lines.extend(
            self.selected_skills
                .iter()
                .map(|name| format!("[Skill: {name}]")),
        );
        if !self.text.is_empty() {
            lines.push(self.text.clone());
        }
        lines.extend(
            self.attachments
                .iter()
                .map(UserImageAttachment::placeholder_summary),
        );
        lines.join("\n")
    }
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

#[cfg(test)]
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

#[cfg(test)]
fn is_runtime_context_section(text: &str) -> bool {
    matches!(
        text.lines().next(),
        Some(
            "[Context: Hard Context]"
                | "[Context: Pinned Context]"
                | "[Context: Index]"
                | "[Context: Open Detail]"
        )
    )
}

fn successful_subagent_reconciliation(event: &ToolFinishedEvent) -> Option<(String, String)> {
    if event.name != "agent__reconcile" || event.outcome != ToolOutcome::Success {
        return None;
    }

    let output = serde_json::from_str::<serde_json::Value>(event.output.as_deref()?).ok()?;
    if output.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }
    let data = output.get("data")?;
    if data.get("reconciled").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }

    Some((
        data.get("run_id")?.as_str()?.to_string(),
        data.get("child_session_id")?.as_str()?.to_string(),
    ))
}

fn append_streaming_tool_output(
    tool: &mut ToolView,
    stream: crate::tool::ToolOutputStream,
    chunk: &str,
) {
    let mut data = tool
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .and_then(|value| value.get("data").cloned().or(Some(value)))
        .unwrap_or_else(|| serde_json::json!({}));

    let key = stream.as_str();
    let existing = data
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    data[key] = serde_json::Value::String(format!("{existing}{chunk}"));
    data[stream.truncated_key()] = serde_json::Value::Bool(false);
    data["streaming"] = serde_json::Value::Bool(true);

    tool.output = Some(
        serde_json::json!({
            "ok": true,
            "tool": tool.name,
            "data": data,
        })
        .to_string(),
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionView {
    pub call_id: String,
    pub tool_name: String,
    pub summary: String,
    pub arguments: Option<String>,
    pub rationale: Option<String>,
    pub origin_label: Option<String>,
    pub can_allow_always: bool,
    pub grant_summary: Option<String>,
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
            can_allow_always: event.can_allow_always,
            grant_summary: event.grant_summary,
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

#[cfg(test)]
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompactionView {
    pub summary: String,
    pub streaming: bool,
}

#[derive(Debug, Clone)]
pub struct Timeline {
    items: Vec<TimelineItem>,
    revisions: Vec<u64>,
    next_revision: u64,
    mutation_revision: u64,
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
            mutation_revision: 0,
            cache_id: next_timeline_cache_id(),
        }
    }
}

impl Timeline {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn from_conversation(messages: Vec<ConversationMessage>) -> Self {
        let mut timeline = Self::new();
        for message in messages {
            match message.role {
                ConversationRole::User => timeline.push_item(TimelineItem::User(MessageView {
                    id: None,
                    submission_id: None,
                    role: MessageRole::User,
                    text: message.content,
                    attachments: Vec::new(),
                    selected_skills: Vec::new(),
                    streaming: false,
                    queued: false,
                })),
                ConversationRole::Summary if is_runtime_context_section(&message.content) => {}
                ConversationRole::Summary => timeline.push_restored_compaction(message.content),
                ConversationRole::Assistant => {
                    timeline.push_item(TimelineItem::Assistant(MessageView {
                        id: None,
                        submission_id: None,
                        role: MessageRole::Assistant,
                        text: message.content,
                        attachments: Vec::new(),
                        selected_skills: Vec::new(),
                        streaming: false,
                        queued: false,
                    }))
                }
            }
        }
        timeline
    }

    pub fn from_transcript_records(records: &[TranscriptRecord]) -> Self {
        super::transcript_read_model::timeline_from_transcript_records(records)
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

    /// Changes whenever timeline contents or item ordering changes.
    pub fn mutation_revision(&self) -> u64 {
        self.mutation_revision
    }

    fn push_item(&mut self, item: TimelineItem) {
        let revision = self.next_item_revision();
        self.mark_mutated();

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
        if index < self.revisions.len() {
            let revision = self.next_item_revision();
            self.revisions[index] = revision;
            self.mark_mutated();
        }
    }

    fn next_item_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1).max(1);
        revision
    }

    fn mark_mutated(&mut self) {
        self.mutation_revision = self.mutation_revision.wrapping_add(1).max(1);
    }

    fn remove_item(&mut self, index: usize) {
        if index < self.items.len() {
            self.items.remove(index);
            if index < self.revisions.len() {
                self.revisions.remove(index);
            }
            self.mark_mutated();
        }
    }

    pub fn push_user_message(&mut self, event: UserMessageEvent) {
        self.push_item(TimelineItem::User(MessageView {
            id: event.message_id,
            submission_id: Some(event.submission_id),
            role: MessageRole::User,
            text: event.content.text,
            attachments: event.content.attachments,
            selected_skills: event.content.selected_skills,
            streaming: false,
            queued: event.queued,
        }));
    }

    pub(crate) fn push_restored_message(&mut self, role: MessageRole, text: String) {
        self.push_item(match role {
            MessageRole::User => TimelineItem::User(MessageView {
                id: None,
                submission_id: None,
                role,
                text,
                attachments: Vec::new(),
                selected_skills: Vec::new(),
                streaming: false,
                queued: false,
            }),
            MessageRole::Assistant => TimelineItem::Assistant(MessageView {
                id: None,
                submission_id: None,
                role,
                text,
                attachments: Vec::new(),
                selected_skills: Vec::new(),
                streaming: false,
                queued: false,
            }),
        });
    }

    pub(crate) fn push_restored_reasoning(&mut self, item_id: String, text: String) {
        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id,
            text,
            streaming: false,
        }));
    }

    pub fn push_assistant_delta(&mut self, event: AssistantDeltaEvent) {
        if let Some(index) = self.active_assistant_message_index(event.message_id.as_deref()) {
            if let TimelineItem::Assistant(message) = &mut self.items[index] {
                message.text.push_str(&event.delta);
                message.streaming = true;
            }
            self.bump_revision(index);
            return;
        }

        self.push_item(TimelineItem::Assistant(MessageView {
            id: event.message_id,
            submission_id: None,
            role: MessageRole::Assistant,
            text: event.delta,
            attachments: Vec::new(),
            selected_skills: Vec::new(),
            streaming: true,
            queued: false,
        }));
    }

    /// Close every in-flight assistant stream bubble.
    ///
    /// Multi-iteration agent loops (common with chat-completions models such as
    /// Grok) stream assistant text, then tools, then more assistant text.
    /// The session transport only emits `AssistantDone` at turn end, so without closing the
    /// pre-tool bubble, later deltas append into it and the final summary shows
    /// under the user message ahead of tool cards.
    pub fn finalize_all_assistant_messages(&mut self) {
        let indexes = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| match item {
                TimelineItem::Assistant(message) if message.streaming => Some(index),
                _ => None,
            })
            .collect::<Vec<_>>();
        for index in indexes {
            if let TimelineItem::Assistant(message) = &mut self.items[index] {
                message.streaming = false;
            }
            self.bump_revision(index);
        }
    }

    pub fn activate_queued_user_message(&mut self, submission_id: &str) -> bool {
        let Some(index) = self.items.iter().position(|item| {
            matches!(
                item,
                TimelineItem::User(MessageView {
                    submission_id: Some(id),
                    queued: true,
                    ..
                }) if id == submission_id
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
        if self.items.len() != original_len {
            self.mark_mutated();
        }
    }

    pub fn remove_first_queued_user_message_preview(&mut self, submission_id: &str) -> bool {
        let Some(index) = self.items.iter().position(|item| {
            matches!(
                item,
                TimelineItem::User(MessageView {
                    submission_id: Some(id),
                    queued: true,
                    ..
                }) if id == submission_id
            )
        }) else {
            return false;
        };

        self.items.remove(index);
        self.revisions.remove(index);
        self.mark_mutated();
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
        let reconciliation = successful_subagent_reconciliation(&event);
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
        } else {
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

        if let Some((run_id, child_session_id)) = reconciliation {
            self.reconcile_subagent_result(&run_id, &child_session_id);
        }
        true
    }

    fn reconcile_subagent_result(&mut self, run_id: &str, child_session_id: &str) {
        let Some(index) = self.items.iter().position(|item| {
            let TimelineItem::Tool(tool) = item else {
                return false;
            };
            if !is_subagent_tool_name(&tool.name) {
                return false;
            }
            let Some(output) = tool.output.as_deref() else {
                return false;
            };
            let Ok(output) = serde_json::from_str::<serde_json::Value>(output) else {
                return false;
            };
            let Some(data) = output.get("data") else {
                return false;
            };
            data.get("run_id").and_then(serde_json::Value::as_str) == Some(run_id)
                && data
                    .get("child_session_id")
                    .and_then(serde_json::Value::as_str)
                    == Some(child_session_id)
        }) else {
            return;
        };

        if let TimelineItem::Tool(tool) = &mut self.items[index] {
            let Some(output) = tool.output.as_deref() else {
                return;
            };
            let Ok(mut output) = serde_json::from_str::<serde_json::Value>(output) else {
                return;
            };
            let Some(data) = output
                .get_mut("data")
                .and_then(serde_json::Value::as_object_mut)
            else {
                return;
            };
            data.insert("unreconciled".into(), serde_json::Value::Bool(false));
            data.insert("reconciled".into(), serde_json::Value::Bool(true));
            tool.output = Some(output.to_string());
        }
        self.bump_revision(index);
    }

    pub fn push_tool_output_delta(&mut self, event: ToolOutputDeltaEvent) -> bool {
        let Some(index) = self.find_tool_index(&event.call_id) else {
            return false;
        };
        if let TimelineItem::Tool(tool) = &mut self.items[index] {
            if tool.status != ToolExecutionStatus::Running {
                return false;
            }
            append_streaming_tool_output(tool, event.stream, &event.chunk);
        }
        self.bump_revision(index);
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
            tool_name: event.tool_name.unwrap_or_else(|| "unknown tool".into()),
            summary: event.summary.unwrap_or_else(|| {
                "Permission resolved without an earlier prompt in timeline".into()
            }),
            arguments: None,
            rationale: None,
            origin_label: event.origin_label,
            can_allow_always: false,
            grant_summary: None,
            status: match event.decision {
                PermissionDecision::Approved => PermissionPromptStatus::Approved,
                PermissionDecision::Denied => PermissionPromptStatus::Denied,
            },
            resolution_reason: event.reason,
        }));
    }

    pub(crate) fn push_restored_permission_decision(
        &mut self,
        call_id: String,
        tool_name: String,
        summary: String,
        arguments: Option<String>,
        status: PermissionPromptStatus,
        resolution_reason: Option<String>,
        origin_label: Option<String>,
    ) {
        self.push_item(TimelineItem::Permission(PermissionView {
            call_id,
            tool_name,
            summary,
            arguments,
            rationale: None,
            origin_label,
            can_allow_always: false,
            grant_summary: None,
            status,
            resolution_reason,
        }));
    }

    pub fn push_error(&mut self, event: ErrorEvent) {
        self.push_item(TimelineItem::Error(ErrorView {
            message: event.message,
            details: event.details,
        }));
    }

    pub fn push_restored_compaction(&mut self, summary: impl Into<String>) {
        self.push_item(TimelineItem::Compaction(CompactionView {
            summary: summary.into(),
            streaming: false,
        }));
    }

    fn streaming_compaction_index(&self) -> Option<usize> {
        self.items.iter().rposition(|item| {
            matches!(
                item,
                TimelineItem::Compaction(CompactionView {
                    streaming: true,
                    ..
                })
            )
        })
    }

    /// Begin a durable streaming compaction block in the transcript.
    pub fn start_compaction(&mut self) {
        if let Some(index) = self.streaming_compaction_index() {
            self.remove_item(index);
        }
        self.push_item(TimelineItem::Compaction(CompactionView {
            summary: String::new(),
            streaming: true,
        }));
    }

    pub fn append_compaction_preview(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(index) = self.streaming_compaction_index() {
            if let TimelineItem::Compaction(view) = &mut self.items[index] {
                view.summary.push_str(delta);
                self.bump_revision(index);
            }
            return;
        }
        // Late delta without Started: open a streaming block so text is not lost.
        self.push_item(TimelineItem::Compaction(CompactionView {
            summary: delta.to_string(),
            streaming: true,
        }));
    }

    pub fn commit_compaction_with_summary(&mut self, summary: String) {
        if let Some(index) = self.streaming_compaction_index() {
            if let TimelineItem::Compaction(view) = &mut self.items[index] {
                if !summary.is_empty() {
                    view.summary = summary;
                }
                view.streaming = false;
                self.bump_revision(index);
            }
            return;
        }
        if !summary.is_empty() {
            self.push_restored_compaction(summary);
        }
    }

    pub fn finish_compaction(&mut self, committed: bool) {
        if committed {
            if let Some(index) = self.streaming_compaction_index() {
                if let TimelineItem::Compaction(view) = &mut self.items[index] {
                    view.streaming = false;
                    self.bump_revision(index);
                }
                return;
            }
            // Committed with no in-flight block: leave a durable empty block.
            self.push_restored_compaction(String::new());
            return;
        }
        if let Some(index) = self.streaming_compaction_index() {
            self.remove_item(index);
        }
    }

    #[cfg(test)]
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

    pub fn update_active_subagent_tool_live_summary(
        &mut self,
        child_session_id: &str,
        _agent_name: Option<&str>,
        parent_tool_call_id: Option<&str>,
        status: &str,
        summary: &str,
    ) -> bool {
        let Some(parent_tool_call_id) = parent_tool_call_id else {
            return false;
        };
        let index = self.find_tool_index(parent_tool_call_id);
        let Some(index) = index else {
            return false;
        };

        let Some(TimelineItem::Tool(tool)) = self.items.get_mut(index) else {
            return false;
        };
        if !is_subagent_tool_name(&tool.name)
            || !matches!(
                tool.status,
                ToolExecutionStatus::Pending | ToolExecutionStatus::Running
            )
        {
            return false;
        }

        let agent_name = agent_name_for_subagent_tool(&tool.name).unwrap_or(tool.name.as_str());
        tool.summary = summary.to_string();
        tool.output = Some(
            serde_json::json!({
                "status": status,
                "summary": summary,
                "child_session_id": child_session_id,
                "agent_name": agent_name,
            })
            .to_string(),
        );
        self.bump_revision(index);
        true
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

pub(crate) fn restored_tool_summary(name: &str, ok: bool) -> String {
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
    use crate::tool::{ToolOutputStream, ToolResult};
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use serde_json::json;

    #[test]
    fn compaction_preview_accumulates_and_commits_as_durable_block() {
        let mut timeline = Timeline::new();
        timeline.start_compaction();
        timeline.append_compaction_preview("first ");
        timeline.append_compaction_preview("second");

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Compaction(view)] if view.streaming && view.summary == "first second"
        ));

        timeline.finish_compaction(true);
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Compaction(view)] if !view.streaming && view.summary == "first second"
        ));
    }

    #[test]
    fn compaction_preview_is_discarded_on_non_success() {
        let mut timeline = Timeline::new();
        timeline.start_compaction();
        timeline.append_compaction_preview("transient");
        timeline.finish_compaction(false);

        assert!(timeline.items().is_empty());
    }

    #[test]
    fn late_compaction_delta_opens_streaming_block() {
        let mut timeline = Timeline::new();
        timeline.append_compaction_preview("late");
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Compaction(view)] if view.streaming && view.summary == "late"
        ));
    }

    #[test]
    fn restored_conversation_hides_runtime_context_sections_but_keeps_normal_summaries() {
        let timeline = Timeline::from_conversation(vec![
            ConversationMessage {
                role: ConversationRole::Summary,
                content: "[Context: Hard Context]\n- hidden runtime context".into(),
            },
            ConversationMessage {
                role: ConversationRole::Summary,
                content: "Compacted conversation summary".into(),
            },
            ConversationMessage {
                role: ConversationRole::User,
                content: "continue".into(),
            },
        ]);

        assert_eq!(timeline.items().len(), 2);
        assert!(matches!(
            &timeline.items()[0],
            TimelineItem::Compaction(view)
                if !view.streaming && view.summary == "Compacted conversation summary"
        ));
        assert!(matches!(
            &timeline.items()[1],
            TimelineItem::User(message) if message.text == "continue"
        ));
    }

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
    fn multi_iteration_assistant_text_appears_after_tools_not_under_user() {
        // Repro for Grok/chat-completions multi-iteration turns: without
        // finalizing the first assistant bubble when tools start, the final
        // summary deltas append into the pre-tool message.
        let mut timeline = Timeline::new();
        timeline.push_user_message(UserMessageEvent::new("do the task"));
        timeline.push_assistant_delta(AssistantDeltaEvent::new("planning…"));
        assert!(timeline.push_tool_started(ToolStartedEvent::new("c1", "shell__exec", "run",)));
        // Simulate production: finalize open assistant streams when tools begin.
        timeline.finalize_all_assistant_messages();
        timeline.push_assistant_delta(AssistantDeltaEvent::new("final summary"));
        timeline.finalize_assistant_message(None);

        let items = timeline.items();
        assert_eq!(items.len(), 4);
        assert!(matches!(&items[0], TimelineItem::User(m) if m.text == "do the task"));
        assert!(matches!(
            &items[1],
            TimelineItem::Assistant(m) if m.text == "planning…" && !m.streaming
        ));
        assert!(matches!(&items[2], TimelineItem::Tool(_)));
        assert!(matches!(
            &items[3],
            TimelineItem::Assistant(m) if m.text == "final summary" && !m.streaming
        ));
    }

    #[test]
    fn shell_tool_output_deltas_update_running_tool_output() {
        let mut timeline = Timeline::new();

        timeline.push_tool_started(ToolStartedEvent {
            call_id: "tool-stream".into(),
            name: "shell__exec".into(),
            summary: "run script".into(),
            arguments: Some(json!({"command":"script"}).to_string()),
        });
        assert!(timeline.push_tool_output_delta(ToolOutputDeltaEvent::new(
            "tool-stream",
            ToolOutputStream::Stdout,
            "one\n",
        )));
        assert!(timeline.push_tool_output_delta(ToolOutputDeltaEvent::new(
            "tool-stream",
            ToolOutputStream::Stderr,
            "warn\n",
        )));

        let items = timeline.items();
        match &items[0] {
            TimelineItem::Tool(tool) => {
                assert_eq!(tool.status, ToolExecutionStatus::Running);
                let output = tool.output.as_deref().expect("streaming output");
                let value: serde_json::Value = serde_json::from_str(output).expect("json output");
                let data = value.get("data").expect("data");
                assert_eq!(data["stdout"], json!("one\n"));
                assert_eq!(data["stderr"], json!("warn\n"));
                assert_eq!(data["streaming"], json!(true));
            }
            other => panic!("expected tool item, got {other:?}"),
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

    fn successful_reconciliation_output(run_id: &str, child_session_id: &str) -> String {
        json!({
            "ok": true,
            "tool": "agent__reconcile",
            "data": {
                "run_id": run_id,
                "child_session_id": child_session_id,
                "reconciled": true,
            }
        })
        .to_string()
    }

    #[test]
    fn successful_reconcile_updates_matching_subagent_card_by_stable_ids() {
        let mut timeline = Timeline::new();
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "subagent-1".into(),
            name: "agent__fixer".into(),
            summary: "completed".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__fixer",
                    "data": {
                        "run_id": "run-1",
                        "child_session_id": "child-1",
                        "unreconciled": true,
                        "reconciled": false,
                    }
                })
                .to_string(),
            ),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "subagent-2".into(),
            name: "agent__fixer".into(),
            summary: "completed".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__fixer",
                    "data": {
                        "run_id": "run-2",
                        "child_session_id": "child-2",
                        "unreconciled": true,
                        "reconciled": false,
                    }
                })
                .to_string(),
            ),
        });

        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "reconcile-1".into(),
            name: "agent__reconcile".into(),
            summary: "reconciled".into(),
            outcome: ToolOutcome::Success,
            output: Some(successful_reconciliation_output("run-1", "child-1")),
        });

        let subagent_data = |index| match &timeline.items()[index] {
            TimelineItem::Tool(tool) => serde_json::from_str::<serde_json::Value>(
                tool.output.as_deref().expect("subagent output"),
            )
            .expect("valid subagent output")["data"]
                .clone(),
            other => panic!("expected tool item, got {other:?}"),
        };
        assert_eq!(subagent_data(0)["unreconciled"], json!(false));
        assert_eq!(subagent_data(0)["reconciled"], json!(true));
        assert_eq!(subagent_data(1)["unreconciled"], json!(true));
        assert_eq!(subagent_data(1)["reconciled"], json!(false));
    }

    #[test]
    fn failed_or_invalid_reconcile_does_not_update_subagent_card() {
        let mut timeline = Timeline::new();
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "subagent-1".into(),
            name: "agent__fixer".into(),
            summary: "completed".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__fixer",
                    "data": {
                        "run_id": "run-1",
                        "child_session_id": "child-1",
                        "unreconciled": true,
                        "reconciled": false,
                    }
                })
                .to_string(),
            ),
        });

        for (call_id, outcome, output) in [
            (
                "reconcile-failed",
                ToolOutcome::Failure,
                successful_reconciliation_output("run-1", "child-1"),
            ),
            (
                "reconcile-invalid",
                ToolOutcome::Success,
                json!({"ok": true, "data": {"reconciled": true}}).to_string(),
            ),
        ] {
            timeline.push_tool_finished(ToolFinishedEvent {
                call_id: call_id.into(),
                name: "agent__reconcile".into(),
                summary: "reconcile".into(),
                outcome,
                output: Some(output),
            });
        }

        let TimelineItem::Tool(tool) = &timeline.items()[0] else {
            panic!("expected subagent tool item");
        };
        let output = serde_json::from_str::<serde_json::Value>(
            tool.output.as_deref().expect("subagent output"),
        )
        .expect("valid subagent output");
        assert_eq!(output["data"]["unreconciled"], json!(true));
        assert_eq!(output["data"]["reconciled"], json!(false));
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
            can_allow_always: false,
            grant_summary: None,
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
    fn auto_review_resolution_creates_reviewer_permission_card() {
        let mut timeline = Timeline::new();
        timeline.resolve_permission(PermissionResolutionEvent {
            call_id: "call-auto".into(),
            decision: PermissionDecision::Approved,
            reason: Some("safe write".into()),
            tool_name: Some("fs__write".into()),
            summary: Some("fs__write a.txt".into()),
            origin_label: Some("reviewer".into()),
        });

        match timeline.items().first() {
            Some(TimelineItem::Permission(permission)) => {
                assert_eq!(permission.tool_name, "fs__write");
                assert_eq!(permission.origin_label.as_deref(), Some("reviewer"));
                assert_eq!(permission.status, PermissionPromptStatus::Approved);
                assert_eq!(permission.resolution_reason.as_deref(), Some("safe write"));
            }
            other => panic!("expected auto-review permission card, got {other:?}"),
        }
    }

    #[test]
    fn transcript_restore_preserves_tool_arguments_and_output_for_visible_cards() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
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
                context_branch_id: None,
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
    fn transcript_restore_uses_one_committed_compaction_separator() {
        let records = vec![TranscriptRecord {
            session_id: "session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ContextCompaction(
                crate::agent::ContextCompactionEvent::succeeded("目标\n- 继续任务", 2),
            ),
        }];

        let timeline = Timeline::from_transcript_records(&records);
        let items = timeline.items();

        assert_eq!(items.len(), 1);
        assert!(matches!(
            &items[0],
            TimelineItem::Compaction(view)
                if !view.streaming && view.summary == "目标\n- 继续任务"
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
        }));
        timeline.push_todo_snapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: TodoStatus::Completed,
        }]));
        timeline.apply_auto_continue_changed(AutoContinueChangedEvent::new(AutoContinueState {
            enabled: true,
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
        timeline.push_restored_compaction("prior context folded");

        let items = timeline.items();
        assert!(
            matches!(items.last(), Some(TimelineItem::User(message)) if message.text == "follow up" && message.queued)
        );
        assert!(matches!(
            items.get(items.len() - 2),
            Some(TimelineItem::Compaction(view)) if !view.streaming && view.summary == "prior context folded"
        ));
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
                context_branch_id: None,
                event: TranscriptEvent::AutoContinueChanged {
                    state: AutoContinueState { enabled: true },
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
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
                context_branch_id: None,
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
            context_branch_id: None,
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
    fn transcript_restore_renders_permission_decision_item() {
        let records = vec![TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
                reviewer: None,
                approval: None,
                risk: None,
                reviewer_child_session_id: None,
            },
        }];

        let timeline = Timeline::from_transcript_records(&records);
        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Permission(permission))
                if permission.call_id == "call-1"
                    && permission.tool_name == "shell__exec"
                    && permission.status == PermissionPromptStatus::Denied
                    && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
        ));
    }

    #[test]
    fn transcript_restore_renders_cancelled_tool_call_as_terminal_tool() {
        let records = vec![TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallCancelled {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
            },
        }];

        let timeline = Timeline::from_transcript_records(&records);
        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool))
                if tool.call_id == "call-1"
                    && tool.name == "shell__exec"
                    && tool.status == ToolExecutionStatus::Cancelled
        ));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn transcript_restore_ignores_turn_audit_events() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "hello".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
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
                context_branch_id: None,
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
                context_branch_id: None,
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
                context_branch_id: None,
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
                context_branch_id: None,
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
                context_branch_id: None,
                event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
            },
        ];

        let timeline = Timeline::from_transcript_records(&records);
        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
        assert_eq!(timeline.items().len(), 1);
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

    #[test]
    fn active_subagent_tool_live_summary_updates_existing_parent_card() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "parent-call",
            "agent__explore",
            "inspect src/tui",
        ));

        assert!(timeline.update_active_subagent_tool_live_summary(
            "child-session-1234567890",
            Some("explorer"),
            Some("parent-call"),
            "running",
            "shell__exec — cargo test --lib"
        ));

        let tool = timeline
            .items()
            .iter()
            .find_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .expect("subagent tool exists");
        assert_eq!(tool.summary, "shell__exec — cargo test --lib");
        assert_eq!(tool.status, ToolExecutionStatus::Running);
        let output = tool.output.as_deref().expect("live summary payload exists");
        assert!(output.contains("child-session-1234567890"), "{output}");
        assert!(
            output.contains("shell__exec — cargo test --lib"),
            "{output}"
        );
        assert!(output.contains("explorer"), "{output}");
    }

    #[test]
    fn child_live_summary_without_agent_name_does_not_guess_latest_parent_card() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "explorer-call",
            "agent__explore",
            "inspect src/tui",
        ));
        timeline.push_tool_started(ToolStartedEvent::new(
            "fixer-call",
            "agent__fixer",
            "fix src/tui",
        ));

        assert!(!timeline.update_active_subagent_tool_live_summary(
            "explorer-child",
            None,
            None,
            "running",
            "explorer working",
        ));

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools[0].summary, "inspect src/tui");
        assert_eq!(tools[1].summary, "fix src/tui");
        assert!(tools.iter().all(|tool| tool.output.is_none()));
    }

    #[test]
    fn child_live_summaries_remain_bound_to_their_parent_tool_cards() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "explorer-call",
            "agent__explore",
            "inspect src/tui",
        ));
        timeline.push_tool_started(ToolStartedEvent::new(
            "fixer-call",
            "agent__fixer",
            "fix src/tui",
        ));
        // Events may arrive in a different order from the parent tool cards.
        assert!(timeline.update_active_subagent_tool_live_summary(
            "explorer-child",
            Some("explorer"),
            Some("explorer-call"),
            "running",
            "explorer working",
        ));
        assert!(timeline.update_active_subagent_tool_live_summary(
            "fixer-child",
            Some("fixer"),
            Some("fixer-call"),
            "running",
            "fixer working",
        ));
        assert!(timeline.update_active_subagent_tool_live_summary(
            "explorer-child",
            Some("explorer"),
            Some("explorer-call"),
            "completed",
            "explorer done",
        ));

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools[0].summary, "explorer done");
        assert!(
            tools[0]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("explorer-child"))
        );
        assert_eq!(tools[1].summary, "fixer working");
        assert!(
            tools[1]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("fixer-child"))
        );
    }

    #[test]
    fn duplicate_role_child_events_use_exact_parent_call_id() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "older-explorer-call",
            "agent__explore",
            "inspect older path",
        ));
        timeline.push_tool_started(ToolStartedEvent::new(
            "newer-explorer-call",
            "agent__explore",
            "inspect newer path",
        ));

        assert!(timeline.update_active_subagent_tool_live_summary(
            "older-child",
            Some("explorer"),
            Some("older-explorer-call"),
            "running",
            "older explorer working",
        ));
        assert!(timeline.update_active_subagent_tool_live_summary(
            "newer-child",
            Some("explorer"),
            Some("newer-explorer-call"),
            "running",
            "newer explorer working",
        ));

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools[0].summary, "older explorer working");
        assert_eq!(tools[1].summary, "newer explorer working");
        assert!(
            tools[0]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("older-child"))
        );
        assert!(
            tools[1]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("newer-child"))
        );
    }

    #[test]
    fn late_child_event_does_not_rebind_after_parent_tool_finishes() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "old-parent-call",
            "agent__explore",
            "inspect old path",
        ));
        timeline.push_tool_finished(ToolFinishedEvent::new(
            "old-parent-call",
            "agent__explore",
            "old explorer completed",
            ToolOutcome::Success,
        ));
        timeline.push_tool_started(ToolStartedEvent::new(
            "new-parent-call",
            "agent__explore",
            "inspect new path",
        ));

        assert!(!timeline.update_active_subagent_tool_live_summary(
            "old-child",
            Some("explorer"),
            Some("old-parent-call"),
            "running",
            "late old event",
        ));

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools[0].summary, "old explorer completed");
        assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
        assert_eq!(tools[1].summary, "inspect new path");
        assert!(tools[1].output.is_none());
    }
}
