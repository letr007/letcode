#![allow(dead_code)]

use crate::runtime_context::{RuntimeFrameId, RuntimeFrameProvenance};
use crate::user_content::UserMessageContent;
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProtocolFrameKind {
    ContextSummary,
    User,
    InternalContinuation,
    Assistant,
    AssistantToolCalls,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProtocolItem {
    ContextSummary {
        text: String,
    },
    UserMessage {
        content: UserMessageContent,
    },
    InternalContinuation {
        text: String,
    },
    AssistantText {
        text: String,
    },
    AssistantToolCalls {
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        calls: Vec<ProtocolToolCall>,
    },
    ToolOutput {
        call_id: String,
        output_json: String,
    },
}

impl ProtocolItem {
    pub fn context_summary(text: impl Into<String>) -> Self {
        Self::ContextSummary { text: text.into() }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::UserMessage {
            content: UserMessageContent::new(text, Vec::new()),
        }
    }

    pub fn user_content(content: UserMessageContent) -> Self {
        Self::UserMessage { content }
    }

    pub fn internal_continuation(text: impl Into<String>) -> Self {
        Self::InternalContinuation { text: text.into() }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::AssistantText { text: text.into() }
    }
}

pub(crate) type ProtocolFrameItem = ProtocolItem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtocolFrame {
    /// The durable runtime identity when this frame was projected from a snapshot.
    /// Compatibility history projections deliberately leave this unset.
    pub runtime_frame_id: Option<RuntimeFrameId>,
    /// Provenance is retained for runtime material that is rendered without being
    /// part of the transcript protocol authority.
    pub source_provenance: Option<RuntimeFrameProvenance>,
    pub history_index: usize,
    pub item: ProtocolFrameItem,
}

impl ProtocolFrame {
    pub(crate) fn derived(item: ProtocolFrameItem) -> Self {
        Self {
            runtime_frame_id: None,
            source_provenance: None,
            history_index: usize::MAX,
            item,
        }
    }

    pub(crate) fn from_history_item(history_index: usize, item: &ProtocolItem) -> Self {
        Self {
            runtime_frame_id: None,
            source_provenance: None,
            history_index,
            item: item.clone(),
        }
    }

    pub(crate) fn kind(&self) -> ProtocolFrameKind {
        match self.item {
            ProtocolFrameItem::ContextSummary { .. } => ProtocolFrameKind::ContextSummary,
            ProtocolFrameItem::UserMessage { .. } => ProtocolFrameKind::User,
            ProtocolFrameItem::InternalContinuation { .. } => {
                ProtocolFrameKind::InternalContinuation
            }
            ProtocolFrameItem::AssistantText { .. } => ProtocolFrameKind::Assistant,
            ProtocolFrameItem::AssistantToolCalls { .. } => ProtocolFrameKind::AssistantToolCalls,
            ProtocolFrameItem::ToolOutput { .. } => ProtocolFrameKind::Tool,
        }
    }

    pub(crate) fn stable_prompt_key(&self) -> String {
        match &self.item {
            ProtocolFrameItem::ContextSummary { text } => {
                format!("context_summary:{}:{}", self.history_index, text)
            }
            ProtocolFrameItem::UserMessage { content } => {
                format!("user:{}:{}", self.history_index, content.prompt_plan_text())
            }
            ProtocolFrameItem::InternalContinuation { text } => {
                format!("internal_continuation:{}:{}", self.history_index, text)
            }
            ProtocolFrameItem::AssistantText { text } => {
                format!("assistant:{}:{}", self.history_index, text)
            }
            ProtocolFrameItem::AssistantToolCalls {
                text,
                reasoning_content,
                calls,
            } => format!(
                "assistant_tool_calls:{}:{}:{}:{}",
                self.history_index,
                text.clone().unwrap_or_default(),
                reasoning_content.clone().unwrap_or_default(),
                calls
                    .iter()
                    .map(|call| format!("{}:{}:{}", call.call_id, call.name, call.arguments_json))
                    .collect::<Vec<_>>()
                    .join("|")
            ),
            ProtocolFrameItem::ToolOutput {
                call_id,
                output_json,
            } => format!(
                "tool_output:{}:{}:{}",
                self.history_index, call_id, output_json
            ),
        }
    }

    pub(crate) fn prompt_source_label(&self) -> &'static str {
        match self.kind() {
            ProtocolFrameKind::ContextSummary => "context_summary",
            ProtocolFrameKind::User => "user",
            ProtocolFrameKind::InternalContinuation => "internal_continuation",
            ProtocolFrameKind::Assistant => "assistant",
            ProtocolFrameKind::AssistantToolCalls => "assistant_tool_calls",
            ProtocolFrameKind::Tool => "tool",
        }
    }

    pub(crate) fn to_history_item(&self) -> ProtocolItem {
        self.item.clone()
    }
}

pub(crate) fn history_items_to_frames(history: &[ProtocolItem]) -> Vec<ProtocolFrame> {
    history
        .iter()
        .enumerate()
        .map(|(index, item)| ProtocolFrame::from_history_item(index, item))
        .collect()
}

pub(crate) fn history_items_from_frames(frames: &[ProtocolFrame]) -> Vec<ProtocolItem> {
    frames.iter().map(ProtocolFrame::to_history_item).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallGroupStatus {
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ToolCallGroupProtection {
    pub current_turn: bool,
    pub incomplete: bool,
}

impl ToolCallGroupProtection {
    pub(crate) fn is_protected(self) -> bool {
        self.current_turn || self.incomplete
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolCallGroup {
    pub assistant_index: usize,
    pub tool_output_indexes: Vec<usize>,
    pub call_ids: Vec<String>,
    pub status: ToolCallGroupStatus,
    pub protection: ToolCallGroupProtection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtocolTranscript {
    pub frames: Vec<ProtocolFrame>,
    pub tool_call_groups: Vec<ToolCallGroup>,
}

impl ProtocolTranscript {
    pub(crate) fn protected_history_indexes(&self) -> BTreeSet<usize> {
        let mut protected = BTreeSet::new();
        for group in &self.tool_call_groups {
            if !group.protection.is_protected() {
                continue;
            }
            protected.insert(group.assistant_index);
            protected.extend(group.tool_output_indexes.iter().copied());
        }
        protected
    }

    pub(crate) fn has_incomplete_tool_call_groups(&self) -> bool {
        self.tool_call_groups
            .iter()
            .any(|group| group.status == ToolCallGroupStatus::Incomplete)
    }
}

pub(crate) fn analyze_history_items(
    history: &[ProtocolItem],
    current_turn_start_index: Option<usize>,
) -> Result<ProtocolTranscript> {
    let frames = history
        .iter()
        .enumerate()
        .map(|(index, item)| ProtocolFrame::from_history_item(index, item))
        .collect::<Vec<_>>();
    let current_turn_start_index = current_turn_start_index.unwrap_or(history.len());

    let mut tool_call_groups = Vec::new();
    let mut pending_group: Option<PendingToolCallGroup> = None;

    for frame in &frames {
        match &frame.item {
            ProtocolFrameItem::AssistantToolCalls { text: _, calls, .. } => {
                if let Some(group) = pending_group.take() {
                    tool_call_groups.push(group.finish_incomplete());
                }

                if calls.is_empty() {
                    continue;
                }

                ensure_unique_call_ids(frame.history_index, calls)?;
                pending_group = Some(PendingToolCallGroup::new(
                    frame.history_index,
                    calls.iter().map(|call| call.call_id.clone()).collect(),
                    frame.history_index >= current_turn_start_index,
                ));
            }
            ProtocolFrameItem::ToolOutput { call_id, .. } => {
                let Some(group) = pending_group.as_mut() else {
                    bail!(
                        "orphan tool output at history index {} for call_id '{}'",
                        frame.history_index,
                        call_id
                    );
                };

                group.push_tool_output(frame.history_index, call_id, current_turn_start_index)?;
                if group.pending_call_ids.is_empty() {
                    let group = pending_group.take().expect("pending group should exist");
                    tool_call_groups.push(group.finish_complete());
                }
            }
            _ => {
                if let Some(group) = pending_group.take() {
                    tool_call_groups.push(group.finish_incomplete());
                }
            }
        }
    }

    if let Some(group) = pending_group.take() {
        tool_call_groups.push(group.finish_incomplete());
    }

    Ok(ProtocolTranscript {
        frames,
        tool_call_groups,
    })
}

pub(crate) fn validate_history_items_complete(
    history: &[ProtocolItem],
    current_turn_start_index: Option<usize>,
) -> Result<ProtocolTranscript> {
    let transcript = analyze_history_items(history, current_turn_start_index)?;
    if let Some(group) = transcript
        .tool_call_groups
        .iter()
        .find(|group| group.status == ToolCallGroupStatus::Incomplete)
    {
        bail!(
            "dangling assistant tool calls at history index {} for call_ids {:?}",
            group.assistant_index,
            group.call_ids
        );
    }
    Ok(transcript)
}

/// Compaction can only split history between complete tool-call groups. Moving
/// a boundary left is lossless: the group joins the summarized prefix instead
/// of leaving outputs whose declaring assistant frame was removed.
pub(crate) fn canonical_compaction_boundary(
    history: &[ProtocolItem],
    requested_boundary: usize,
) -> Result<usize> {
    let transcript = analyze_history_items(history, None)?;
    let mut boundary = requested_boundary.min(history.len());

    for group in transcript.tool_call_groups {
        let group_end = group
            .tool_output_indexes
            .last()
            .copied()
            .map(|index| index + 1)
            .unwrap_or(group.assistant_index + 1);
        if group.assistant_index < boundary && boundary < group_end {
            boundary = group.assistant_index;
        }
    }
    Ok(boundary)
}

fn ensure_unique_call_ids(history_index: usize, calls: &[ProtocolToolCall]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for call in calls {
        ensure!(
            seen.insert(call.call_id.clone()),
            "duplicate assistant tool call_id '{}' at history index {}",
            call.call_id,
            history_index
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PendingToolCallGroup {
    assistant_index: usize,
    call_ids: Vec<String>,
    pending_call_ids: BTreeSet<String>,
    tool_output_indexes: Vec<usize>,
    current_turn: bool,
}

impl PendingToolCallGroup {
    fn new(assistant_index: usize, call_ids: Vec<String>, current_turn: bool) -> Self {
        Self {
            assistant_index,
            pending_call_ids: call_ids.iter().cloned().collect(),
            call_ids,
            tool_output_indexes: Vec::new(),
            current_turn,
        }
    }

    fn push_tool_output(
        &mut self,
        history_index: usize,
        call_id: &str,
        current_turn_start_index: usize,
    ) -> Result<()> {
        ensure!(
            self.pending_call_ids.remove(call_id),
            "orphan tool output at history index {} for call_id '{}'",
            history_index,
            call_id
        );
        self.tool_output_indexes.push(history_index);
        if history_index >= current_turn_start_index {
            self.current_turn = true;
        }
        Ok(())
    }

    fn finish_complete(self) -> ToolCallGroup {
        ToolCallGroup {
            assistant_index: self.assistant_index,
            tool_output_indexes: self.tool_output_indexes,
            call_ids: self.call_ids,
            status: ToolCallGroupStatus::Complete,
            protection: ToolCallGroupProtection {
                current_turn: self.current_turn,
                incomplete: false,
            },
        }
    }

    fn finish_incomplete(self) -> ToolCallGroup {
        ToolCallGroup {
            assistant_index: self.assistant_index,
            tool_output_indexes: self.tool_output_indexes,
            call_ids: self.call_ids,
            status: ToolCallGroupStatus::Incomplete,
            protection: ToolCallGroupProtection {
                current_turn: self.current_turn,
                incomplete: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type HistoryItem = ProtocolItem;
    type HistoryToolCall = ProtocolToolCall;

    fn tool_call(call_id: &str) -> HistoryToolCall {
        HistoryToolCall {
            call_id: call_id.into(),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"src/main.rs"}"#.into(),
        }
    }

    #[test]
    fn compaction_boundary_keeps_a_complete_tool_call_batch_atomic() {
        let history = vec![
            HistoryItem::user("question"),
            HistoryItem::AssistantToolCalls {
                text: None,
                reasoning_content: None,
                calls: vec![tool_call("call-1"), tool_call("call-2")],
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
            },
            HistoryItem::ToolOutput {
                call_id: "call-2".into(),
                output_json: "{}".into(),
            },
            HistoryItem::assistant("done"),
        ];

        assert_eq!(canonical_compaction_boundary(&history, 0).unwrap(), 0);
        assert_eq!(canonical_compaction_boundary(&history, 1).unwrap(), 1);
        assert_eq!(canonical_compaction_boundary(&history, 2).unwrap(), 1);
        assert_eq!(canonical_compaction_boundary(&history, 3).unwrap(), 1);
        assert_eq!(canonical_compaction_boundary(&history, 4).unwrap(), 4);
        assert_eq!(canonical_compaction_boundary(&history, 5).unwrap(), 5);
    }

    #[test]
    fn validation_rejects_orphan_tool_output() {
        let history = vec![
            HistoryItem::user("question"),
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
            },
        ];

        let error = validate_history_items_complete(&history, Some(0))
            .expect_err("orphan tool output must fail");
        assert!(error.to_string().contains("orphan tool output"));
    }

    #[test]
    fn analysis_marks_incomplete_group_as_protected() {
        let history = vec![
            HistoryItem::user("question"),
            HistoryItem::AssistantToolCalls {
                text: Some("working".into()),
                reasoning_content: None,
                calls: vec![tool_call("call-1")],
            },
        ];

        let transcript = analyze_history_items(&history, Some(1)).expect("analysis succeeds");

        assert!(transcript.has_incomplete_tool_call_groups());
        assert_eq!(transcript.tool_call_groups.len(), 1);
        let group = &transcript.tool_call_groups[0];
        assert_eq!(group.status, ToolCallGroupStatus::Incomplete);
        assert!(group.protection.current_turn);
        assert!(group.protection.incomplete);
        assert_eq!(transcript.protected_history_indexes(), BTreeSet::from([1]));
    }

    #[test]
    fn validation_rejects_dangling_assistant_tool_calls() {
        let history = vec![
            HistoryItem::user("question"),
            HistoryItem::AssistantToolCalls {
                text: None,
                reasoning_content: None,
                calls: vec![tool_call("call-1")],
            },
        ];

        let error = validate_history_items_complete(&history, Some(1))
            .expect_err("dangling assistant tool calls must fail");
        assert!(error.to_string().contains("dangling assistant tool calls"));
    }
}
