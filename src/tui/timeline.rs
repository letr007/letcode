use super::events::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, PermissionDecision,
    PermissionRequestEvent, PermissionResolutionEvent, ReasoningDeltaEvent, TodoSnapshotEvent,
    ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent,
    UserMessageEvent,
};
use crate::agent::{
    AutoContinueState, TodoItem, agent_name_for_subagent_tool, is_subagent_tool_name,
    subagent_tool_name_for_agent_name,
};
#[cfg(test)]
use crate::agent::{ConversationMessage, ConversationRole};
use crate::transcript::TranscriptRecord;
use crate::user_content::UserImageAttachment;

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineItem {
    User(MessageView),
    Delegation(DelegationView),
    Reasoning(ReasoningView),
    Assistant(MessageView),
    Tool(ToolView),
    Todo(TodoView),
    Permission(PermissionView),
    AutoReview(AutoReviewDecisionView),
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
            Self::AutoReview(decision) => vec![DisplayBlock::StatusLine {
                label: "auto-review".into(),
                text: format!(
                    "{} · {} · {}",
                    if decision.allowed {
                        "approved"
                    } else {
                        "denied"
                    },
                    decision.tool_name,
                    decision.approval,
                ),
            }],
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
    pub started_at: Option<std::time::Instant>,
    pub duration_ms: Option<u64>,
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
struct SubagentToolBinding {
    call_id: String,
    run_id: String,
    child_session_id: String,
    agent_name: String,
    task: String,
    background: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SettledSubagentWait {
    call_id: String,
    child_session_id: String,
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
pub struct AutoReviewDecisionView {
    pub call_id: String,
    pub tool_name: String,
    pub approval: String,
    pub risk: Option<String>,
    pub rationale: String,
    pub allowed: bool,
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
    last_reasoning_tick_second: Option<u64>,
    tool_indices: HashMap<String, usize>,
    permission_indices: HashMap<String, usize>,
    subagent_tools_by_run: HashMap<String, SubagentToolBinding>,
    subagent_run_by_child: HashMap<String, String>,
    foreground_waits: HashMap<String, String>,
    settled_subagent_waits: HashMap<String, SettledSubagentWait>,
    pending_subagent_starts: Vec<SubagentToolBinding>,
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
            last_reasoning_tick_second: None,
            tool_indices: HashMap::new(),
            permission_indices: HashMap::new(),
            subagent_tools_by_run: HashMap::new(),
            subagent_run_by_child: HashMap::new(),
            foreground_waits: HashMap::new(),
            settled_subagent_waits: HashMap::new(),
            pending_subagent_starts: Vec::new(),
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
        } else if let Some(index) = self.items.iter().position(is_queued_user_item) {
            self.items.insert(index, item);
            self.revisions.insert(index, revision);
        } else {
            self.items.push(item);
            self.revisions.push(revision);
        }
        self.rebuild_lookup_indices();
    }

    fn rebuild_lookup_indices(&mut self) {
        self.tool_indices.clear();
        self.permission_indices.clear();
        for (index, item) in self.items.iter().enumerate() {
            match item {
                TimelineItem::Tool(tool) => {
                    self.tool_indices
                        .entry(tool.call_id.clone())
                        .or_insert(index);
                }
                TimelineItem::Permission(permission) => {
                    self.permission_indices
                        .entry(permission.call_id.clone())
                        .or_insert(index);
                }
                _ => {}
            }
        }
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
            self.rebuild_lookup_indices();
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

    pub(crate) fn push_restored_reasoning(
        &mut self,
        item_id: String,
        text: String,
        duration_ms: Option<u64>,
    ) {
        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id,
            text,
            streaming: false,
            started_at: None,
            duration_ms,
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
        self.rebuild_lookup_indices();
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
        self.rebuild_lookup_indices();
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

        let previous_reasoning_index = self
            .items
            .iter()
            .position(is_queued_user_item)
            .unwrap_or(self.items.len())
            .checked_sub(1)
            .filter(|index| matches!(self.items[*index], TimelineItem::Reasoning(_)));
        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id: event.item_id,
            text: event.delta,
            streaming: true,
            started_at: Some(event.observed_at),
            duration_ms: None,
        }));
        if let Some(index) = previous_reasoning_index {
            self.bump_revision(index);
        }
        self.last_reasoning_tick_second = Some(0);
    }

    pub fn tick_reasoning_elapsed(&mut self, now: std::time::Instant) {
        const TICK_MS: u128 = 100;
        let current_tick = self
            .items
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Reasoning(reasoning) if reasoning.streaming => {
                    reasoning.started_at.map(|started_at| {
                        let elapsed_ms = now.saturating_duration_since(started_at).as_millis();
                        if elapsed_ms < 1_000 {
                            elapsed_ms / TICK_MS
                        } else {
                            10 + elapsed_ms / 1_000
                        }
                    })
                }
                _ => None,
            })
            .max()
            .map(|tick| u64::try_from(tick).unwrap_or(u64::MAX));
        if current_tick.is_none() {
            self.last_reasoning_tick_second = None;
            return;
        }
        if current_tick == self.last_reasoning_tick_second {
            return;
        }
        self.last_reasoning_tick_second = current_tick;
        let indices = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                matches!(item, TimelineItem::Reasoning(reasoning) if reasoning.streaming)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        for index in indices {
            self.bump_revision(index);
        }
    }

    pub fn push_delegation(&mut self, agent_name: impl Into<String>, task: impl Into<String>) {
        self.push_item(TimelineItem::Delegation(DelegationView {
            agent_name: agent_name.into(),
            task: task.into(),
        }));
    }

    pub fn finalize_reasoning(&mut self, event: crate::tui::events::ReasoningDoneEvent) {
        if let Some(index) = self.active_reasoning_index(&event.item_id) {
            if let TimelineItem::Reasoning(reasoning) = &mut self.items[index] {
                reasoning.text = event.text.clone();
                reasoning.streaming = false;
                reasoning.duration_ms = reasoning.started_at.map(|started_at| {
                    u64::try_from(
                        event
                            .observed_at
                            .saturating_duration_since(started_at)
                            .as_millis(),
                    )
                    .unwrap_or(u64::MAX)
                });
                reasoning.started_at = None;
            }
            self.bump_revision(index);
            if !self.has_streaming_reasoning() {
                self.last_reasoning_tick_second = None;
            }
            return;
        }

        let previous_reasoning_index = self
            .items
            .iter()
            .position(is_queued_user_item)
            .unwrap_or(self.items.len())
            .checked_sub(1)
            .filter(|index| matches!(self.items[*index], TimelineItem::Reasoning(_)));
        self.push_item(TimelineItem::Reasoning(ReasoningView {
            item_id: event.item_id,
            text: event.text,
            streaming: false,
            started_at: None,
            duration_ms: None,
        }));
        if let Some(index) = previous_reasoning_index {
            self.bump_revision(index);
        }
        if !self.has_streaming_reasoning() {
            self.last_reasoning_tick_second = None;
        }
    }

    pub fn seal_active_reasoning(&mut self, observed_at: std::time::Instant) {
        let indices = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                matches!(item, TimelineItem::Reasoning(reasoning) if reasoning.streaming)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        for index in indices {
            if let TimelineItem::Reasoning(reasoning) = &mut self.items[index] {
                reasoning.duration_ms = reasoning.started_at.map(|started_at| {
                    u64::try_from(
                        observed_at
                            .saturating_duration_since(started_at)
                            .as_millis(),
                    )
                    .unwrap_or(u64::MAX)
                });
                reasoning.started_at = None;
                reasoning.streaming = false;
            }
            self.bump_revision(index);
        }
        self.last_reasoning_tick_second = None;
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
        if is_subagent_tool_name(&event.name) {
            let agent_name = agent_name_for_subagent_tool(&event.name)
                .unwrap_or(event.name.as_str())
                .to_string();
            let arguments = event
                .arguments
                .as_deref()
                .and_then(|arguments| serde_json::from_str::<serde_json::Value>(arguments).ok());
            let task = arguments
                .as_ref()
                .and_then(|args| args.get("task"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| event.summary.clone());
            let background = arguments
                .as_ref()
                .and_then(|args| args.get("background"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.pending_subagent_starts
                .retain(|binding| binding.call_id != event.call_id);
            self.pending_subagent_starts.push(SubagentToolBinding {
                call_id: event.call_id.clone(),
                run_id: String::new(),
                child_session_id: String::new(),
                agent_name,
                task,
                background,
            });
        }
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
        if event.name == crate::tool_names::TOOL_AGENT_WAIT {
            return self.push_subagent_wait_finished(event);
        }

        let finished_subagent_call_id =
            is_subagent_tool_name(&event.name).then(|| event.call_id.clone());
        let terminal_subagent_result = if is_subagent_tool_name(&event.name) {
            event
                .output
                .as_deref()
                .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
                .and_then(|value| value.get("data").cloned().or(Some(value)))
                .and_then(|data| subagent_summary_from_wait_data(&data))
        } else {
            None
        };
        if let Some(call_id) = finished_subagent_call_id.as_deref() {
            self.pending_subagent_starts
                .retain(|binding| binding.call_id != call_id);
        }
        if let Some(result) = terminal_subagent_result {
            if !self.subagent_tools_by_run.contains_key(&result.run_id) {
                let task = self
                    .find_tool_index(&event.call_id)
                    .and_then(|index| self.items.get(index))
                    .and_then(|item| match item {
                        TimelineItem::Tool(tool) => tool
                            .arguments
                            .as_deref()
                            .and_then(|arguments| {
                                serde_json::from_str::<serde_json::Value>(arguments).ok()
                            })
                            .and_then(|args| {
                                args.get("task")
                                    .and_then(serde_json::Value::as_str)
                                    .map(str::to_string)
                            }),
                        _ => None,
                    })
                    .unwrap_or_else(|| event.summary.clone());
                self.subagent_run_by_child
                    .insert(result.child_session_id.clone(), result.run_id.clone());
                self.subagent_tools_by_run.insert(
                    result.run_id.clone(),
                    SubagentToolBinding {
                        call_id: event.call_id.clone(),
                        run_id: result.run_id.clone(),
                        child_session_id: result.child_session_id.clone(),
                        agent_name: result.agent_name.clone(),
                        task,
                        background: false,
                    },
                );
            }
            return self.finish_subagent_tool(&result, false);
        }

        let subagent_binding = if is_subagent_tool_name(&event.name) {
            event
                .output
                .as_deref()
                .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
                .and_then(|value| value.get("data").cloned().or(Some(value)))
                .and_then(|data| {
                    let run_id = data.get("run_id")?.as_str()?.to_string();
                    let child_session_id = data.get("child_session_id")?.as_str()?.to_string();
                    let agent_name = data
                        .get("agent_name")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                        .or_else(|| {
                            agent_name_for_subagent_tool(&event.name).map(str::to_string)
                        })?;
                    let background = data
                        .get("background")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    (data.get("active").and_then(serde_json::Value::as_bool) == Some(true))
                        .then_some((run_id, child_session_id, agent_name, background))
                })
        } else {
            None
        };

        let call_id = event.call_id.clone();
        if !self.push_ordinary_tool_finished(event) {
            return false;
        }

        if let Some((run_id, child_session_id, agent_name, background)) = subagent_binding {
            let task = self
                .find_tool_index(&call_id)
                .and_then(|index| self.items.get(index))
                .and_then(|item| match item {
                    TimelineItem::Tool(tool) => tool
                        .arguments
                        .as_deref()
                        .and_then(|arguments| {
                            serde_json::from_str::<serde_json::Value>(arguments).ok()
                        })
                        .and_then(|args| {
                            args.get("task")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_string)
                        }),
                    _ => None,
                })
                .unwrap_or_else(|| "subagent running".into());
            self.subagent_run_by_child
                .insert(child_session_id.clone(), run_id.clone());
            self.subagent_tools_by_run.insert(
                run_id.clone(),
                SubagentToolBinding {
                    call_id: call_id.clone(),
                    run_id,
                    child_session_id,
                    agent_name,
                    task,
                    background,
                },
            );
        }

        true
    }

    fn push_subagent_wait_finished(&mut self, event: ToolFinishedEvent) -> bool {
        let result = event
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .and_then(|value| value.get("data").cloned().or(Some(value)))
            .and_then(|data| subagent_summary_from_wait_data(&data));
        let handled = result
            .as_ref()
            .is_some_and(|result| self.finish_subagent_wait(&event.call_id, result));
        if handled {
            return true;
        }
        let foreground_run_id = self
            .foreground_waits
            .iter()
            .find_map(|(run_id, call_id)| (call_id == &event.call_id).then(|| run_id.clone()));
        if self.find_tool_index(&event.call_id).is_some() {
            let finished = self.push_ordinary_tool_finished(event);
            if let Some(run_id) = foreground_run_id {
                self.foreground_waits.remove(&run_id);
            }
            return finished;
        }
        self.remove_tool(&event.call_id);
        if let Some(run_id) = foreground_run_id {
            self.foreground_waits.remove(&run_id);
        }
        false
    }

    fn push_ordinary_tool_finished(&mut self, event: ToolFinishedEvent) -> bool {
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
        true
    }

    pub fn remove_tool(&mut self, call_id: &str) -> bool {
        let Some(index) = self.find_tool_index(call_id) else {
            return false;
        };
        self.remove_item(index);
        true
    }

    pub fn cancel_foreground_subagent_wait(&mut self, wait_call_id: &str) -> bool {
        let Some(run_id) = self
            .foreground_waits
            .iter()
            .find_map(|(run_id, call_id)| (call_id == wait_call_id).then(|| run_id.clone()))
        else {
            return false;
        };
        self.foreground_waits.remove(&run_id);
        let Some(binding) = self.subagent_tools_by_run.get(&run_id) else {
            return false;
        };
        let Some(index) = self.find_tool_index(wait_call_id) else {
            return false;
        };
        let Some(TimelineItem::Tool(tool)) = self.items.get_mut(index) else {
            return false;
        };
        tool.status = ToolExecutionStatus::Cancelled;
        tool.summary = "subagent wait cancelled".into();
        tool.output = serde_json::to_string(&serde_json::json!({
            "run_id": binding.run_id,
            "child_session_id": binding.child_session_id,
            "agent_name": binding.agent_name,
            "status": "cancelled",
            "summary": tool.summary,
            "waiting": false,
            "active": false,
            "background": false,
        }))
        .ok();
        self.bump_revision(index);
        true
    }

    pub fn cancel_foreground_subagent_waits(&mut self) -> usize {
        let waits = self.foreground_waits.drain().collect::<Vec<_>>();
        let mut cancelled = 0usize;
        for (run_id, wait_call_id) in waits {
            let Some(binding) = self.subagent_tools_by_run.get(&run_id) else {
                continue;
            };
            let Some(index) = self.find_tool_index(&wait_call_id) else {
                continue;
            };
            let Some(TimelineItem::Tool(tool)) = self.items.get_mut(index) else {
                continue;
            };
            tool.status = ToolExecutionStatus::Cancelled;
            tool.summary = "subagent wait cancelled".into();
            tool.output = serde_json::to_string(&serde_json::json!({
                "run_id": binding.run_id,
                "child_session_id": binding.child_session_id,
                "agent_name": binding.agent_name,
                "status": "cancelled",
                "summary": tool.summary,
                "waiting": false,
                "active": false,
                "background": false,
            }))
            .ok();
            self.bump_revision(index);
            cancelled = cancelled.saturating_add(1);
        }
        cancelled
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
        if is_subagent_tool_name(name) {
            self.pending_subagent_starts
                .retain(|binding| binding.call_id != call_id);
        }
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
        if event.origin_label.as_deref() == Some("reviewer") {
            self.push_auto_review_decision(AutoReviewDecisionView {
                call_id: event.call_id,
                tool_name: event.tool_name.unwrap_or_else(|| "unknown tool".into()),
                approval: event.approval.unwrap_or_else(|| match event.decision {
                    PermissionDecision::Approved => "once".into(),
                    PermissionDecision::Denied => "deny".into(),
                }),
                risk: event.risk,
                rationale: event.reason.unwrap_or_else(|| "no rationale".into()),
                allowed: matches!(event.decision, PermissionDecision::Approved),
            });
            return;
        }

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

    pub(crate) fn push_auto_review_decision(&mut self, decision: AutoReviewDecisionView) {
        if let Some(index) = self.items.iter().position(|item| {
            matches!(item, TimelineItem::AutoReview(existing) if existing.call_id == decision.call_id)
        }) {
            self.items[index] = TimelineItem::AutoReview(decision);
            self.bump_revision(index);
            return;
        }
        self.push_item(TimelineItem::AutoReview(decision));
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
        approval: Option<String>,
        risk: Option<String>,
        reviewer_child_session_id: Option<String>,
    ) {
        if origin_label.as_deref() == Some("reviewer") {
            let _ = reviewer_child_session_id;
            self.push_auto_review_decision(AutoReviewDecisionView {
                call_id,
                tool_name,
                approval: approval.unwrap_or_else(|| match status {
                    PermissionPromptStatus::Approved => "once".into(),
                    PermissionPromptStatus::Denied => "deny".into(),
                    PermissionPromptStatus::Pending => "pending".into(),
                }),
                risk,
                rationale: resolution_reason.unwrap_or_else(|| "no rationale".into()),
                allowed: status == PermissionPromptStatus::Approved,
            });
            return;
        }

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
        agent_name: Option<&str>,
        parent_tool_call_id: Option<&str>,
        status: &str,
        summary: &str,
    ) -> bool {
        let run_id = self.subagent_run_by_child.get(child_session_id).cloned();
        let target_call_id = run_id
            .as_ref()
            .and_then(|run_id| self.foreground_waits.get(run_id))
            .cloned()
            .or_else(|| {
                run_id
                    .as_ref()
                    .and_then(|run_id| self.subagent_tools_by_run.get(run_id))
                    .map(|binding| binding.call_id.clone())
            })
            .or_else(|| parent_tool_call_id.map(str::to_string));
        let Some(target_call_id) = target_call_id else {
            return false;
        };
        let Some(index) = self.find_tool_index(&target_call_id) else {
            return false;
        };

        let Some(TimelineItem::Tool(tool)) = self.items.get_mut(index) else {
            return false;
        };
        let waiting = run_id
            .as_ref()
            .is_some_and(|run_id| self.foreground_waits.contains_key(run_id));
        let background = if waiting {
            false
        } else {
            run_id
                .as_ref()
                .and_then(|run_id| self.subagent_tools_by_run.get(run_id))
                .map(|binding| binding.background)
                .or_else(|| {
                    tool.arguments
                        .as_deref()
                        .and_then(|arguments| {
                            serde_json::from_str::<serde_json::Value>(arguments).ok()
                        })
                        .and_then(|args| {
                            args.get("background").and_then(serde_json::Value::as_bool)
                        })
                })
                .unwrap_or(false)
        };
        if !is_subagent_tool_name(&tool.name)
            || (!waiting
                && !matches!(
                    tool.status,
                    ToolExecutionStatus::Pending | ToolExecutionStatus::Running
                ))
        {
            return false;
        }

        let resolved_agent_name = agent_name
            .or_else(|| {
                run_id
                    .as_ref()
                    .and_then(|run_id| self.subagent_tools_by_run.get(run_id))
                    .map(|binding| binding.agent_name.as_str())
            })
            .or_else(|| agent_name_for_subagent_tool(&tool.name))
            .unwrap_or(tool.name.as_str());
        tool.summary = summary.to_string();
        tool.output = Some(
            serde_json::json!({
                "run_id": run_id,
                "status": status,
                "summary": summary,
                "child_session_id": child_session_id,
                "agent_name": resolved_agent_name,
                "waiting": waiting,
                "active": true,
                "background": background,
            })
            .to_string(),
        );
        if waiting {
            tool.status = ToolExecutionStatus::Running;
        }
        self.bump_revision(index);
        true
    }

    pub fn register_subagent_started(
        &mut self,
        run_id: &str,
        child_session_id: &str,
        agent_name: &str,
        task: &str,
    ) -> bool {
        if self.subagent_tools_by_run.contains_key(run_id) {
            return true;
        }
        let position = self
            .pending_subagent_starts
            .iter()
            .position(|binding| binding.agent_name == agent_name && binding.task == task)
            .or_else(|| {
                self.pending_subagent_starts
                    .iter()
                    .position(|binding| binding.agent_name == agent_name)
            });
        let Some(position) = position else {
            return false;
        };
        let mut binding = self.pending_subagent_starts.remove(position);
        binding.run_id = run_id.to_string();
        binding.child_session_id = child_session_id.to_string();
        self.subagent_run_by_child
            .insert(child_session_id.to_string(), run_id.to_string());
        self.subagent_tools_by_run
            .insert(run_id.to_string(), binding);
        true
    }

    pub fn begin_subagent_wait(&mut self, wait_call_id: &str, run_id: &str) -> bool {
        if self.foreground_waits.contains_key(run_id) {
            return false;
        }
        let Some(binding) = self.subagent_tools_by_run.get(run_id).cloned() else {
            return false;
        };
        let Some(index) = self.find_tool_index(&binding.call_id) else {
            return false;
        };
        if !matches!(
            self.items.get(index),
            Some(TimelineItem::Tool(tool))
                if is_subagent_tool_name(&tool.name)
                    && tool
                        .output
                        .as_deref()
                        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
                        .and_then(|value| {
                            value
                                .get("data")
                                .unwrap_or(&value)
                                .get("active")
                                .and_then(serde_json::Value::as_bool)
                        })
                        == Some(true)
        ) {
            return false;
        }
        let Some(TimelineItem::Tool(origin)) = self.items.get(index) else {
            return false;
        };
        let mut waiting = origin.clone();
        waiting.call_id = wait_call_id.to_string();
        waiting.name = subagent_tool_name_for_agent_name(&binding.agent_name)
            .unwrap_or(origin.name.as_str())
            .to_string();
        waiting.status = ToolExecutionStatus::Running;
        waiting.summary = binding.task;
        waiting.output = serde_json::to_string(&serde_json::json!({
            "run_id": binding.run_id,
            "child_session_id": binding.child_session_id,
            "agent_name": binding.agent_name,
            "status": "running",
            "summary": waiting.summary,
            "waiting": true,
            "active": true,
            "background": false,
        }))
        .ok();
        self.foreground_waits
            .insert(run_id.to_string(), wait_call_id.to_string());
        self.push_item(TimelineItem::Tool(waiting));
        true
    }

    pub fn finish_subagent_wait(
        &mut self,
        wait_call_id: &str,
        result: &crate::subagent::SubagentRunSummary,
    ) -> bool {
        if self
            .foreground_waits
            .get(&result.run_id)
            .map(String::as_str)
            != Some(wait_call_id)
        {
            return false;
        }
        self.foreground_waits.remove(&result.run_id);
        if !self.finish_subagent_tool_by_call_id(wait_call_id, result, false) {
            return false;
        }
        self.settled_subagent_waits.insert(
            result.run_id.clone(),
            SettledSubagentWait {
                call_id: wait_call_id.to_string(),
                child_session_id: result.child_session_id.clone(),
            },
        );
        true
    }

    pub fn finish_background_subagent_tool(
        &mut self,
        parent_tool_call_id: &str,
        result: &crate::subagent::SubagentRunSummary,
    ) -> bool {
        if self.foreground_waits.contains_key(&result.run_id)
            || self
                .settled_subagent_waits
                .get(&result.run_id)
                .is_some_and(|wait| {
                    wait.child_session_id == result.child_session_id
                        && self.find_tool_index(&wait.call_id).is_some()
                })
        {
            return false;
        }
        if self
            .subagent_tools_by_run
            .get(&result.run_id)
            .and_then(|binding| self.find_tool_index(&binding.call_id))
            .and_then(|index| self.items.get(index))
            .is_some_and(|item| {
                matches!(
                    item,
                    TimelineItem::Tool(tool) if tool.status == ToolExecutionStatus::Cancelled
                )
            })
        {
            return false;
        }
        if !self.subagent_tools_by_run.contains_key(&result.run_id) {
            let Some(index) = self.find_tool_index(parent_tool_call_id) else {
                return false;
            };
            let Some(TimelineItem::Tool(tool)) = self.items.get(index) else {
                return false;
            };
            if !is_subagent_tool_name(&tool.name) {
                return false;
            }
            self.subagent_tools_by_run.insert(
                result.run_id.clone(),
                SubagentToolBinding {
                    call_id: parent_tool_call_id.to_string(),
                    run_id: result.run_id.clone(),
                    child_session_id: result.child_session_id.clone(),
                    agent_name: result.agent_name.clone(),
                    task: tool.summary.clone(),
                    background: true,
                },
            );
            self.subagent_run_by_child
                .insert(result.child_session_id.clone(), result.run_id.clone());
        }
        self.finish_subagent_tool(result, true)
    }

    fn finish_subagent_tool(
        &mut self,
        result: &crate::subagent::SubagentRunSummary,
        background: bool,
    ) -> bool {
        let Some(binding) = self.subagent_tools_by_run.get(&result.run_id).cloned() else {
            return false;
        };
        self.finish_subagent_tool_by_call_id(&binding.call_id, result, background)
    }

    fn finish_subagent_tool_by_call_id(
        &mut self,
        call_id: &str,
        result: &crate::subagent::SubagentRunSummary,
        background: bool,
    ) -> bool {
        let Some(index) = self.find_tool_index(call_id) else {
            return false;
        };
        let Some(TimelineItem::Tool(tool)) = self.items.get_mut(index) else {
            return false;
        };
        tool.status = match result.status {
            crate::subagent::SubagentStatus::Completed => ToolExecutionStatus::Succeeded,
            crate::subagent::SubagentStatus::Cancelled => ToolExecutionStatus::Cancelled,
            crate::subagent::SubagentStatus::Running
            | crate::subagent::SubagentStatus::Failed
            | crate::subagent::SubagentStatus::BudgetExhausted
            | crate::subagent::SubagentStatus::TimedOut => ToolExecutionStatus::Failed,
        };
        tool.summary = result.summary.clone();
        tool.output = serde_json::to_string(&serde_json::json!({
            "run_id": result.run_id,
            "child_session_id": result.child_session_id,
            "agent_name": result.agent_name,
            "status": result.status.as_str(),
            "summary": result.summary,
            "structured_result": result.structured_result,
            "active": false,
            "background": background,
            "waiting": false,
        }))
        .ok();
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
        self.tool_indices.get(call_id).copied()
    }

    fn find_permission_index(&self, call_id: &str) -> Option<usize> {
        self.permission_indices.get(call_id).copied()
    }

    fn has_streaming_reasoning(&self) -> bool {
        self.items
            .iter()
            .any(|item| matches!(item, TimelineItem::Reasoning(reasoning) if reasoning.streaming))
    }

    fn active_reasoning_index(&self, item_id: &str) -> Option<usize> {
        self.items.iter().rposition(|item| match item {
            TimelineItem::Reasoning(reasoning) => {
                reasoning.streaming && reasoning.item_id == item_id
            }
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

pub(crate) fn subagent_summary_from_wait_data(
    data: &serde_json::Value,
) -> Option<crate::subagent::SubagentRunSummary> {
    let run_id = data.get("run_id")?.as_str()?.to_string();
    let child_session_id = data.get("child_session_id")?.as_str()?.to_string();
    let agent_name = data.get("agent_name")?.as_str()?.to_string();
    let status = match data.get("status")?.as_str()? {
        "completed" => crate::subagent::SubagentStatus::Completed,
        "failed" => crate::subagent::SubagentStatus::Failed,
        "budget_exhausted" => crate::subagent::SubagentStatus::BudgetExhausted,
        "cancelled" => crate::subagent::SubagentStatus::Cancelled,
        "timed_out" => crate::subagent::SubagentStatus::TimedOut,
        _ => return None,
    };
    let failure_kind = match data.get("failure_kind").and_then(serde_json::Value::as_str) {
        Some("hard") => Some(crate::subagent::SubagentFailureKind::Hard),
        Some("logical") => Some(crate::subagent::SubagentFailureKind::Logical),
        _ => None,
    };
    let summary = data
        .get("summary")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let structured_result = data.get("structured_result").cloned().and_then(|value| {
        serde_json::from_value::<crate::subagent::StructuredSubagentResult>(value).ok()
    })?;
    Some(crate::subagent::SubagentRunSummary {
        run_id,
        child_session_id,
        agent_name,
        status,
        failure_kind,
        summary,
        structured_result,
    })
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
    fn reasoning_elapsed_ticks_at_subsecond_then_second_boundaries() {
        let start = std::time::Instant::now();
        let mut timeline = Timeline::new();
        timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "Inspecting", start));
        let initial_revision = timeline.item_revisions()[0];

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(99));
        assert_eq!(timeline.item_revisions()[0], initial_revision);

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(100));
        let subsecond_revision = timeline.item_revisions()[0];
        assert_ne!(subsecond_revision, initial_revision);

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(999));
        let last_subsecond_revision = timeline.item_revisions()[0];
        assert_ne!(last_subsecond_revision, subsecond_revision);

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(1_500));
        let first_second_revision = timeline.item_revisions()[0];
        assert_ne!(first_second_revision, last_subsecond_revision);

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(1_999));
        assert_eq!(timeline.item_revisions()[0], first_second_revision);

        timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(2_000));
        assert_ne!(timeline.item_revisions()[0], first_second_revision);
    }

    #[test]
    fn reasoning_done_without_delta_refreshes_preceding_compact_item() {
        let start = std::time::Instant::now();
        let mut timeline = Timeline::new();
        timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "First", start));
        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reasoning-1",
            "First",
            start + std::time::Duration::from_millis(100),
        ));
        let preceding_revision = timeline.item_revisions()[0];

        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reasoning-without-delta",
            "Second",
            start + std::time::Duration::from_millis(200),
        ));

        assert_ne!(timeline.item_revisions()[0], preceding_revision);
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
                if first.text == "First" && second.text == "Second"
        ));
    }

    #[test]
    fn reused_reasoning_item_id_finalizes_the_latest_streaming_item() {
        let start = std::time::Instant::now();
        let mut timeline = Timeline::new();
        timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reused", "First draft", start));
        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reused",
            "First final",
            start + std::time::Duration::from_millis(100),
        ));
        timeline.push_reasoning_delta(ReasoningDeltaEvent::at(
            "reused",
            "Second draft",
            start + std::time::Duration::from_millis(200),
        ));
        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reused",
            "Second final",
            start + std::time::Duration::from_millis(500),
        ));

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
                if first.text == "First final"
                    && first.duration_ms == Some(100)
                    && !first.streaming
                    && second.text == "Second final"
                    && second.duration_ms == Some(300)
                    && !second.streaming
        ));
    }

    #[test]
    fn reasoning_done_records_duration_and_missing_start_stays_unknown() {
        let start = std::time::Instant::now();
        let mut timeline = Timeline::new();
        timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "Draft", start));
        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reasoning-1",
            "Final",
            start + std::time::Duration::from_millis(725),
        ));
        timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
            "reasoning-without-delta",
            "Recovered",
            start + std::time::Duration::from_secs(1),
        ));

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
                if !first.streaming
                    && first.started_at.is_none()
                    && first.duration_ms == Some(725)
                    && first.text == "Final"
                    && !second.streaming
                    && second.started_at.is_none()
                    && second.duration_ms.is_none()
        ));
    }

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
    fn reviewer_permission_resolutions_keep_event_order() {
        let mut timeline = Timeline::new();
        timeline.resolve_permission(PermissionResolutionEvent {
            call_id: "call-1".into(),
            decision: PermissionDecision::Approved,
            reason: Some("safe".into()),
            tool_name: Some("fs__write".into()),
            summary: Some("write file".into()),
            origin_label: Some("reviewer".into()),
            approval: Some("once".into()),
            risk: Some("low".into()),
            reviewer_child_session_id: Some("reviewer-a".into()),
        });
        timeline.push_tool_started(ToolStartedEvent::new(
            "tool-between",
            "fs__read",
            "read file",
        ));
        timeline.resolve_permission(PermissionResolutionEvent {
            call_id: "call-2".into(),
            decision: PermissionDecision::Denied,
            reason: Some("blocked".into()),
            tool_name: Some("shell__exec".into()),
            summary: Some("run command".into()),
            origin_label: Some("reviewer".into()),
            approval: Some("deny".into()),
            risk: Some("high".into()),
            reviewer_child_session_id: Some("reviewer-a".into()),
        });

        assert!(matches!(
            timeline.items(),
            [TimelineItem::AutoReview(first), TimelineItem::Tool(_), TimelineItem::AutoReview(second)]
                if first.call_id == "call-1"
                    && first.allowed
                    && second.call_id == "call-2"
                    && !second.allowed
        ));
    }

    #[test]
    fn reviewer_permission_resolution_updates_matching_call_in_place() {
        let mut timeline = Timeline::new();
        let resolution = |decision, reason: &str| PermissionResolutionEvent {
            call_id: "call-1".into(),
            decision,
            reason: Some(reason.into()),
            tool_name: Some("shell__exec".into()),
            summary: Some("run command".into()),
            origin_label: Some("reviewer".into()),
            approval: Some(if decision == PermissionDecision::Approved {
                "once".into()
            } else {
                "deny".into()
            }),
            risk: Some("high".into()),
            reviewer_child_session_id: None,
        };
        timeline.resolve_permission(resolution(PermissionDecision::Approved, "safe"));
        timeline.resolve_permission(resolution(PermissionDecision::Denied, "updated"));

        assert!(matches!(
            timeline.items(),
            [TimelineItem::AutoReview(decision)]
                if decision.call_id == "call-1"
                    && !decision.allowed
                    && decision.rationale == "updated"
        ));
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
        let explorer_data = tools[0]
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .expect("explorer live data");
        assert_eq!(
            explorer_data
                .get("background")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
        assert_eq!(tools[1].summary, "fixer working");
        let fixer_data = tools[1]
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .expect("fixer live data");
        assert_eq!(
            fixer_data
                .get("background")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn background_child_live_summary_keeps_background_flag() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "inspect in background".into(),
            arguments: Some(json!({"task": "inspect", "background": true}).to_string()),
        });

        assert!(timeline.update_active_subagent_tool_live_summary(
            "background-child",
            Some("explorer"),
            Some("background-call"),
            "running",
            "background explorer working",
        ));

        let Some(TimelineItem::Tool(tool)) = timeline.items().first() else {
            panic!("expected background subagent tool");
        };
        let data = tool
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .expect("background live data");
        assert_eq!(
            data.get("background").and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn waiting_run_clones_a_live_card_below_the_background_receipt() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        assert!(matches!(
            timeline.items().last(),
            Some(TimelineItem::Tool(waiting))
                if waiting.call_id == "wait-call"
                    && waiting.status == ToolExecutionStatus::Running
        ));
        assert!(timeline.update_active_subagent_tool_live_summary(
            "child-bg",
            Some("explorer"),
            Some("background-call"),
            "running",
            "search__rg — searching wait flow",
        ));

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].call_id, "background-call");
        assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
        assert_eq!(tools[0].summary, "explorer running");
        let background_data = tools[0]
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .expect("background card data");
        assert_eq!(
            background_data
                .get("data")
                .unwrap_or(&background_data)
                .get("background")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(tools[1].call_id, "wait-call");
        assert_eq!(tools[1].name, "agent__explore");
        assert_eq!(tools[1].status, ToolExecutionStatus::Running);
        assert_eq!(tools[1].summary, "search__rg — searching wait flow");
        let data = tools[1]
            .output
            .as_deref()
            .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
            .expect("waiting card data");
        assert_eq!(
            data.get("waiting").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            data.get("background").and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn waiting_run_completes_the_cloned_wait_card() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        let result = crate::subagent::SubagentRunSummary {
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            agent_name: "explorer".into(),
            status: crate::subagent::SubagentStatus::Completed,
            failure_kind: None,
            summary: "wait flow verified".into(),
            structured_result: crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "wait flow verified".into(),
                malformed: false,
                findings: Vec::new(),
                files_read: vec!["src/tui/timeline.rs".into()],
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-bg".into(),
                child_session_id: "child-bg".into(),
                raw_excerpt: None,
            },
        };
        let output = ToolResult::ok(
            crate::tool_names::TOOL_AGENT_WAIT,
            json!({
                "run_id": result.run_id,
                "child_session_id": result.child_session_id,
                "agent_name": result.agent_name,
                "status": result.status.as_str(),
                "failure_kind": serde_json::Value::Null,
                "summary": result.summary,
                "structured_result": result.structured_result,
                "active": false
            }),
        );

        assert!(
            timeline.push_tool_finished(ToolFinishedEvent {
                call_id: "wait-call".into(),
                name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                summary: "explorer completed".into(),
                outcome: ToolOutcome::Success,
                output: Some(
                    serde_json::to_value(output)
                        .expect("serialize output")
                        .to_string()
                ),
            })
        );

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].call_id, "background-call");
        assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
        assert_eq!(tools[0].summary, "explorer running");
        assert!(
            tools[0]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("\"background\":true"))
        );
        assert_eq!(tools[1].call_id, "wait-call");
        assert_eq!(tools[1].status, ToolExecutionStatus::Succeeded);
        assert_eq!(tools[1].summary, "wait flow verified");
        assert!(
            tools[1]
                .output
                .as_deref()
                .is_some_and(|output| output.contains("\"waiting\":false"))
        );
    }

    #[test]
    fn interrupted_wait_stays_terminal_after_late_child_activity() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        assert_eq!(timeline.cancel_foreground_subagent_waits(), 1);
        assert!(!timeline.update_active_subagent_tool_live_summary(
            "child-bg",
            Some("explorer"),
            Some("background-call"),
            "running",
            "late child action",
        ));
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
                if background.call_id == "background-call"
                    && background.status == ToolExecutionStatus::Succeeded
                    && waiting.call_id == "wait-call"
                    && waiting.status == ToolExecutionStatus::Cancelled
                    && waiting.summary == "subagent wait cancelled"
        ));
    }

    #[test]
    fn unstructured_wait_failure_clears_foreground_routing() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        assert!(timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "wait-call".into(),
            name: crate::tool_names::TOOL_AGENT_WAIT.into(),
            summary: "wait failed".into(),
            outcome: ToolOutcome::Failure,
            output: Some(json!({"ok": false, "error": "wait failed"}).to_string()),
        }));
        assert!(!timeline.update_active_subagent_tool_live_summary(
            "child-bg",
            Some("explorer"),
            Some("background-call"),
            "running",
            "late child activity",
        ));
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
                if background.summary == "explorer running"
                    && waiting.call_id == "wait-call"
                    && waiting.status == ToolExecutionStatus::Failed
        ));
    }

    #[test]
    fn late_background_completion_does_not_overwrite_a_completed_wait_clone() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        let result = crate::subagent::SubagentRunSummary {
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            agent_name: "explorer".into(),
            status: crate::subagent::SubagentStatus::Completed,
            failure_kind: None,
            summary: "wait completed".into(),
            structured_result: crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "wait completed".into(),
                malformed: false,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-bg".into(),
                child_session_id: "child-bg".into(),
                raw_excerpt: None,
            },
        };

        assert!(timeline.finish_subagent_wait("wait-call", &result));
        assert!(!timeline.finish_background_subagent_tool("background-call", &result));
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
                if background.summary == "explorer running"
                    && waiting.summary == "wait completed"
        ));
    }

    #[test]
    fn a_second_wait_does_not_replace_the_active_wait_clone() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });

        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        assert!(!timeline.begin_subagent_wait("wait-call-2", "run-bg"));
        assert_eq!(
            timeline
                .items()
                .iter()
                .filter(|item| matches!(item, TimelineItem::Tool(_)))
                .count(),
            2
        );
    }

    #[test]
    fn waiting_run_uses_its_terminal_result_when_completion_events_race() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explore in background".into(),
            arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
        });
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "background-call".into(),
            name: "agent__explore".into(),
            summary: "explorer running".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                json!({
                    "ok": true,
                    "tool": "agent__explore",
                    "data": {
                        "run_id": "run-bg",
                        "child_session_id": "child-bg",
                        "agent_name": "explorer",
                        "status": "running",
                        "summary": "inspect wait flow",
                        "active": true,
                        "background": true
                    }
                })
                .to_string(),
            ),
        });
        assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
        let result = crate::subagent::SubagentRunSummary {
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            agent_name: "explorer".into(),
            status: crate::subagent::SubagentStatus::Completed,
            failure_kind: None,
            summary: "late completion".into(),
            structured_result: crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "late completion".into(),
                malformed: false,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-bg".into(),
                child_session_id: "child-bg".into(),
                raw_excerpt: None,
            },
        };
        assert!(!timeline.finish_background_subagent_tool("background-call", &result));
        assert!(timeline.finish_subagent_wait("wait-call", &result));
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
                if background.call_id == "background-call"
                    && background.status == ToolExecutionStatus::Succeeded
                    && background.summary == "explorer running"
                    && waiting.call_id == "wait-call"
                    && waiting.status == ToolExecutionStatus::Succeeded
                    && waiting.summary == "late completion"
        ));
    }

    #[test]
    fn background_completion_replaces_running_receipt_on_parent_tool() {
        let mut timeline = Timeline::new();
        timeline.push_tool_started(ToolStartedEvent::new(
            "background-call",
            "agent__explore",
            "inspect in background",
        ));
        timeline.push_tool_finished(ToolFinishedEvent::new(
            "background-call",
            "agent__explore",
            "explorer running",
            ToolOutcome::Success,
        ));
        let result = crate::subagent::SubagentRunSummary {
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            agent_name: "explorer".into(),
            status: crate::subagent::SubagentStatus::Completed,
            failure_kind: None,
            summary: "background finding".into(),
            structured_result: crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "background finding".into(),
                malformed: false,
                findings: vec!["found it".into()],
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-bg".into(),
                child_session_id: "child-bg".into(),
                raw_excerpt: None,
            },
        };

        assert!(timeline.finish_background_subagent_tool("background-call", &result));
        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool))
                if tool.status == ToolExecutionStatus::Succeeded
                    && tool.summary == "background finding"
                    && tool.output.as_deref().is_some_and(|output| output.contains("child-bg"))
        ));
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
