#![allow(dead_code)]

use crate::runtime_context::{RuntimeFrameId, RuntimeFrameProvenance};
use crate::user_content::{UserMessageContent, UserMessagePart};
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

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
        /// Provider-native reasoning state needed for exact replay (e.g. signed
        /// Anthropic thinking blocks), stored as JSON. Never rendered as ordinary text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_wire: Option<String>,
        calls: Vec<ProtocolToolCall>,
    },
    ToolOutput {
        call_id: String,
        output_json: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<crate::user_content::UserImageAttachment>,
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
                reasoning_wire,
                calls,
            } => format!(
                "assistant_tool_calls:{}:{}:{}:{}:{}",
                self.history_index,
                text.clone().unwrap_or_default(),
                reasoning_content.clone().unwrap_or_default(),
                reasoning_wire.clone().unwrap_or_default(),
                calls
                    .iter()
                    .map(|call| format!("{}:{}:{}", call.call_id, call.name, call.arguments_json))
                    .collect::<Vec<_>>()
                    .join("|")
            ),
            ProtocolFrameItem::ToolOutput {
                call_id,
                output_json,
                images,
            } => format!(
                "tool_output:{}:{}:{}:{}",
                self.history_index,
                call_id,
                output_json,
                images
                    .iter()
                    .map(crate::user_content::UserImageAttachment::prompt_plan_placeholder)
                    .collect::<Vec<_>>()
                    .join("|")
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

/// Incremental protocol validation state for the live append path. The durable
/// snapshot remains the authority; this cache is rebuilt for restore/compaction
/// and only avoids replaying the whole protocol stream for ordinary appends.
static NEXT_APPEND_LINEAGE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtocolAppendState {
    initialized: bool,
    frontier_token: u64,
    generation: u64,
    frame_ids: Vec<RuntimeFrameId>,
    completed_groups: Vec<IncrementalToolCallGroup>,
    historical_incomplete_groups: Vec<IncrementalToolCallGroup>,
    tail_open_group: Option<IncrementalToolCallGroup>,
    next_group_order: usize,
    current_turn_start_index: Option<usize>,
    protected_frame_ids: Vec<RuntimeFrameId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IncrementalToolCallGroup {
    assistant_index: usize,
    call_ids: Vec<String>,
    pending_call_ids: BTreeSet<String>,
    tool_output_indexes: Vec<usize>,
    frame_ids: Vec<RuntimeFrameId>,
    order: usize,
}

impl ProtocolAppendState {
    pub(crate) fn empty() -> Self {
        Self {
            initialized: false,
            frontier_token: 0,
            generation: 0,
            frame_ids: Vec::new(),
            completed_groups: Vec::new(),
            historical_incomplete_groups: Vec::new(),
            tail_open_group: None,
            next_group_order: 0,
            current_turn_start_index: None,
            protected_frame_ids: Vec::new(),
        }
    }

    pub(crate) fn from_frames(
        frames: &[ProtocolFrame],
        current_turn_start_index: Option<usize>,
    ) -> Result<Self> {
        let mut state = Self::empty();
        state.initialized = true;
        state.frontier_token = NEXT_APPEND_LINEAGE.fetch_add(1, Ordering::Relaxed);
        state.current_turn_start_index = current_turn_start_index;
        for frame in frames {
            let frame_id = frame.runtime_frame_id.ok_or_else(|| {
                anyhow::anyhow!("incremental protocol state requires runtime frame identity")
            })?;
            state.append_internal(
                frame.history_index,
                frame_id,
                &frame.item,
                current_turn_start_index,
                false,
            )?;
        }
        state.recompute_protected_frame_ids();
        Ok(state)
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub(crate) fn frame_count(&self) -> usize {
        self.frame_ids.len()
    }

    pub(crate) fn frontier_token(&self) -> u64 {
        self.frontier_token
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn last_frame_id(&self) -> Option<RuntimeFrameId> {
        self.frame_ids.last().copied()
    }

    pub(crate) fn current_turn_start_index(&self) -> Option<usize> {
        self.current_turn_start_index
    }

    pub(crate) fn appended_frame_ids(
        &self,
        token: u64,
        generation: u64,
        frame_count: usize,
    ) -> Option<&[RuntimeFrameId]> {
        (self.initialized
            && self.frontier_token == token
            && self.generation >= generation
            && self.frame_ids.len() >= frame_count)
            .then_some(&self.frame_ids[frame_count..])
    }

    pub(crate) fn protected_frame_ids(&self) -> &[RuntimeFrameId] {
        &self.protected_frame_ids
    }

    pub(crate) fn has_incomplete_tool_call_groups(&self) -> bool {
        !self.historical_incomplete_groups.is_empty() || self.tail_open_group.is_some()
    }

    pub(crate) fn has_historical_incomplete_tool_call_groups(&self) -> bool {
        !self.historical_incomplete_groups.is_empty()
    }

    pub(crate) fn incomplete_tool_call_ids(&self) -> BTreeSet<String> {
        self.tail_open_group
            .iter()
            .flat_map(|group| group.pending_call_ids.iter().cloned())
            .collect()
    }

    pub(crate) fn append(
        &mut self,
        history_index: usize,
        frame_id: RuntimeFrameId,
        item: &ProtocolFrameItem,
        current_turn_start_index: Option<usize>,
    ) -> Result<()> {
        self.append_internal(
            history_index,
            frame_id,
            item,
            current_turn_start_index,
            true,
        )
    }

    fn append_internal(
        &mut self,
        history_index: usize,
        frame_id: RuntimeFrameId,
        item: &ProtocolFrameItem,
        current_turn_start_index: Option<usize>,
        enforce_incomplete_group_boundary: bool,
    ) -> Result<()> {
        if enforce_incomplete_group_boundary {
            match item {
                ProtocolFrameItem::ToolOutput { call_id, .. } => {
                    let Some(group) = self.tail_open_group.as_ref() else {
                        bail!(
                            "orphan tool output at history index {} for call_id '{}'",
                            history_index,
                            call_id
                        );
                    };
                    ensure!(
                        group.pending_call_ids.contains(call_id),
                        "orphan tool output at history index {} for call_id '{}'",
                        history_index,
                        call_id
                    );
                }
                _ if self.has_incomplete_tool_call_groups() => {
                    bail!(
                        "cannot append {:?} while assistant tool call group is incomplete",
                        item
                    );
                }
                _ => {}
            }
        }

        match item {
            ProtocolFrameItem::AssistantToolCalls { calls, .. } => {
                ensure_unique_call_ids(history_index, calls)?;
            }
            ProtocolFrameItem::ToolOutput { call_id, .. } => {
                ensure!(
                    self.tail_open_group
                        .as_ref()
                        .is_some_and(|group| group.pending_call_ids.contains(call_id)),
                    "orphan tool output at history index {} for call_id '{}'",
                    history_index,
                    call_id
                );
            }
            _ => {}
        }

        self.initialized = true;
        if self.current_turn_start_index != current_turn_start_index {
            self.current_turn_start_index = current_turn_start_index;
            self.generation = self.generation.saturating_add(1);
            self.recompute_protected_frame_ids();
        }

        match item {
            ProtocolFrameItem::AssistantToolCalls { calls, .. } => {
                if let Some(group) = self.tail_open_group.take() {
                    self.historical_incomplete_groups.push(group);
                }
                if !calls.is_empty() {
                    let order = self.next_group_order;
                    self.next_group_order = self.next_group_order.saturating_add(1);
                    self.tail_open_group = Some(IncrementalToolCallGroup {
                        assistant_index: history_index,
                        call_ids: calls.iter().map(|call| call.call_id.clone()).collect(),
                        pending_call_ids: calls.iter().map(|call| call.call_id.clone()).collect(),
                        tool_output_indexes: Vec::new(),
                        frame_ids: vec![frame_id],
                        order,
                    });
                }
            }
            ProtocolFrameItem::ToolOutput { call_id, .. } => {
                let group = self
                    .tail_open_group
                    .as_mut()
                    .expect("tool output was validated against the tail-open group");
                let removed = group.pending_call_ids.remove(call_id);
                debug_assert!(removed, "tool output was validated against its group");
                group.tool_output_indexes.push(history_index);
                group.frame_ids.push(frame_id);
                if group.pending_call_ids.is_empty() {
                    let group = self
                        .tail_open_group
                        .take()
                        .expect("tail-open group should exist");
                    self.completed_groups.push(group);
                }
            }
            _ => {
                if let Some(group) = self.tail_open_group.take() {
                    self.historical_incomplete_groups.push(group);
                }
            }
        }
        self.frame_ids.push(frame_id);
        self.generation = self.generation.saturating_add(1);
        self.update_protected_frame_ids(history_index, frame_id);
        Ok(())
    }

    fn update_protected_frame_ids(&mut self, history_index: usize, frame_id: RuntimeFrameId) {
        let Some(start) = self.current_turn_start_index else {
            return;
        };
        if history_index >= start {
            self.protect(frame_id);
        }
        let ids = self
            .completed_groups
            .iter()
            .chain(self.historical_incomplete_groups.iter())
            .filter(|group| {
                group.assistant_index.ge(&start)
                    || group
                        .tool_output_indexes
                        .iter()
                        .any(|index| *index >= start)
            })
            .flat_map(|group| group.frame_ids.iter().copied())
            .chain(self.tail_open_group.iter().flat_map(|group| {
                group
                    .frame_ids
                    .iter()
                    .copied()
                    .filter(|_| group.assistant_index >= start)
            }))
            .collect::<Vec<_>>();
        for id in ids {
            self.protect(id);
        }
    }

    fn recompute_protected_frame_ids(&mut self) {
        self.protected_frame_ids.clear();
        let Some(start) = self.current_turn_start_index else {
            return;
        };
        let frame_ids = self
            .frame_ids
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, id)| (index >= start).then_some(id))
            .collect::<Vec<_>>();
        for id in frame_ids {
            self.protect(id);
        }
        let group_ids = self
            .completed_groups
            .iter()
            .chain(self.historical_incomplete_groups.iter())
            .chain(self.tail_open_group.iter())
            .filter(|group| {
                group.assistant_index >= start
                    || group
                        .tool_output_indexes
                        .iter()
                        .any(|index| *index >= start)
            })
            .flat_map(|group| group.frame_ids.iter().copied())
            .collect::<Vec<_>>();
        for id in group_ids {
            self.protect(id);
        }
    }

    fn protect(&mut self, id: RuntimeFrameId) {
        if !self.protected_frame_ids.contains(&id) {
            self.protected_frame_ids.push(id);
        }
    }

    pub(crate) fn tool_call_groups(&self) -> Vec<ToolCallGroup> {
        let mut groups = self
            .completed_groups
            .iter()
            .chain(self.historical_incomplete_groups.iter())
            .chain(self.tail_open_group.iter())
            .collect::<Vec<_>>();
        groups.sort_by_key(|group| group.order);
        groups
            .into_iter()
            .map(|group| {
                let current_turn = self.current_turn_start_index.is_some_and(|start| {
                    group.assistant_index >= start
                        || group
                            .tool_output_indexes
                            .iter()
                            .any(|index| *index >= start)
                });
                ToolCallGroup {
                    assistant_index: group.assistant_index,
                    tool_output_indexes: group.tool_output_indexes.clone(),
                    call_ids: group.call_ids.clone(),
                    status: if group.pending_call_ids.is_empty() {
                        ToolCallGroupStatus::Complete
                    } else {
                        ToolCallGroupStatus::Incomplete
                    },
                    protection: ToolCallGroupProtection {
                        current_turn,
                        incomplete: !group.pending_call_ids.is_empty(),
                    },
                }
            })
            .collect()
    }
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
    canonical_compaction_boundary_with_transcript(&transcript, requested_boundary)
}

/// Like [`canonical_compaction_boundary`], but reuses an already-computed
/// transcript. The boundary only depends on tool-call group span indexes, so it
/// is safe to reuse a transcript produced with any turn cursor — avoiding a
/// second full pass that deep-clones every history item into a ProtocolFrame.
pub(crate) fn canonical_compaction_boundary_with_transcript(
    transcript: &ProtocolTranscript,
    requested_boundary: usize,
) -> Result<usize> {
    let mut boundary = requested_boundary.min(transcript.frames.len());
    for group in &transcript.tool_call_groups {
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

/// Cap on the characters of a single string field folded into an identity
/// fingerprint. Enough to stay content-sensitive for real payloads while keeping
/// cost bounded instead of O(raw payload size).
const IDENTITY_FIELD_PREFIX_CHARS: usize = 128;

impl ProtocolFrame {
    /// Bounded, deterministic integrity fingerprint of one frame. Large string
    /// fields (tool outputs, arguments, text, image data URLs) are covered by
    /// byte length + a bounded prefix instead of being fully serialized, so cost
    /// is O(fields + capped prefixes) rather than O(raw payload size). Used only
    /// for process-local consistency comparisons (pressure frontier / usage
    /// anchor), never persisted.
    pub(crate) fn bounded_identity_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(192);
        match &self.runtime_frame_id {
            Some(id) => {
                buf.push(1);
                buf.extend_from_slice(&serde_json::to_vec(id).unwrap_or_default());
            }
            None => buf.push(0),
        }
        buf.extend_from_slice(&(self.history_index as u64).to_le_bytes());
        self.item.write_bounded(&mut buf);
        buf
    }
}

impl ProtocolItem {
    /// Field tag + byte length + capped prefix, streamed in a stable order.
    fn write_bounded(&self, out: &mut Vec<u8>) {
        macro_rules! field {
            ($tag:expr, $s:expr) => {{
                let bytes = $s.as_bytes();
                out.push($tag);
                out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                let cap = bytes.len().min(IDENTITY_FIELD_PREFIX_CHARS);
                out.extend_from_slice(&bytes[..cap]);
            }};
        }
        match self {
            ProtocolItem::ContextSummary { text } => {
                out.push(0x00);
                field!(0x01, text);
            }
            ProtocolItem::UserMessage { content } => {
                out.push(0x01);
                field!(0x02, &content.text);
                out.push(0x0a);
                out.extend_from_slice(&(content.attachments.len() as u64).to_le_bytes());
                out.push(0x0b);
                out.extend_from_slice(&(content.selected_skills.len() as u64).to_le_bytes());
                for skill in &content.selected_skills {
                    field!(0x0c, skill);
                }
                for part in &content.parts {
                    match part {
                        UserMessagePart::Text { text } => field!(0x0d, text),
                        UserMessagePart::Image { attachment } => {
                            let placeholder = attachment.prompt_plan_placeholder();
                            field!(0x0e, &placeholder);
                        }
                    }
                }
            }
            ProtocolItem::InternalContinuation { text } => {
                out.push(0x02);
                field!(0x01, text);
            }
            ProtocolItem::AssistantText { text } => {
                out.push(0x03);
                field!(0x01, text);
            }
            ProtocolItem::AssistantToolCalls {
                text,
                reasoning_content,
                reasoning_wire,
                calls,
            } => {
                out.push(0x04);
                match text {
                    Some(text) => field!(0x01, text),
                    None => out.push(0x00),
                }
                match reasoning_content {
                    Some(text) => field!(0x02, text),
                    None => out.push(0x00),
                }
                match reasoning_wire {
                    Some(json) => field!(0x17, json),
                    None => out.push(0x00),
                }
                out.push(0x0f);
                out.extend_from_slice(&(calls.len() as u64).to_le_bytes());
                for call in calls {
                    field!(0x10, &call.call_id);
                    field!(0x11, &call.name);
                    field!(0x12, &call.arguments_json);
                }
            }
            ProtocolItem::ToolOutput {
                call_id,
                output_json,
                images,
            } => {
                out.push(0x05);
                field!(0x13, call_id);
                field!(0x14, output_json);
                out.push(0x15);
                out.extend_from_slice(&(images.len() as u64).to_le_bytes());
                for image in images {
                    let placeholder = image.prompt_plan_placeholder();
                    field!(0x16, &placeholder);
                }
            }
        }
    }
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
                reasoning_wire: None,
                calls: vec![tool_call("call-1"), tool_call("call-2")],
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            },
            HistoryItem::ToolOutput {
                call_id: "call-2".into(),
                output_json: "{}".into(),
                images: Vec::new(),
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
    fn incremental_append_cases_match_cold_analyzer_for_valid_tail_sequences() {
        let cases = [
            vec![
                HistoryItem::user("question"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    reasoning_content: None,
                    reasoning_wire: None,
                    calls: vec![tool_call("case-1")],
                },
                HistoryItem::ToolOutput {
                    call_id: "case-1".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
            ],
            vec![
                HistoryItem::assistant("before"),
                HistoryItem::AssistantToolCalls {
                    text: Some("tail".into()),
                    reasoning_content: Some("reasoning".into()),
                    reasoning_wire: None,
                    calls: vec![tool_call("case-2")],
                },
                HistoryItem::ToolOutput {
                    call_id: "case-2".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
                HistoryItem::assistant("after"),
            ],
        ];

        for history in cases {
            let full = analyze_history_items(&history, None).expect("full analysis");
            let frames = history_items_to_frames(&history);
            let mut incremental = ProtocolAppendState::empty();
            for (index, frame) in frames.iter().enumerate() {
                incremental
                    .append(
                        index,
                        RuntimeFrameId::from_persisted(index as u64 + 1),
                        &frame.item,
                        None,
                    )
                    .expect("valid incremental append");
            }
            assert_eq!(incremental.tool_call_groups(), full.tool_call_groups);
        }
    }

    #[test]
    fn incremental_append_matches_full_analysis_for_long_tool_history() {
        let mut history = Vec::new();
        for index in 0..128 {
            history.push(HistoryItem::user(format!("question-{index}")));
            history.push(HistoryItem::assistant(format!("answer-{index}")));
        }
        history.extend([
            HistoryItem::AssistantToolCalls {
                text: Some("working".into()),
                reasoning_content: Some("reasoning".into()),
                reasoning_wire: Some(r#"{"signature":"wire"}"#.into()),
                calls: vec![tool_call("call-1"), tool_call("call-2")],
            },
            HistoryItem::ToolOutput {
                call_id: "call-2".into(),
                output_json: r#"{"result":2}"#.into(),
                images: Vec::new(),
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: r#"{"result":1}"#.into(),
                images: Vec::new(),
            },
            HistoryItem::assistant("final"),
        ]);

        let current_turn_start = Some(history.len() - 4);
        let full = analyze_history_items(&history, current_turn_start).expect("full analysis");
        let frames = history_items_to_frames(&history);
        let mut incremental = ProtocolAppendState::empty();
        for (index, frame) in frames.iter().enumerate() {
            incremental
                .append(
                    index,
                    RuntimeFrameId::from_persisted(index as u64 + 1),
                    &frame.item,
                    current_turn_start,
                )
                .expect("incremental append");
        }

        assert_eq!(incremental.frame_count(), history.len());
        assert_eq!(incremental.tool_call_groups(), full.tool_call_groups);
        let mut expected_protected = (history.len() - 4..history.len())
            .map(|index| RuntimeFrameId::from_persisted(index as u64 + 1))
            .collect::<Vec<_>>();
        expected_protected.sort();
        assert_eq!(
            incremental.protected_frame_ids(),
            expected_protected.as_slice()
        );
        assert!(!incremental.has_incomplete_tool_call_groups());
    }

    #[test]
    fn incremental_append_rejects_non_tail_incomplete_groups_and_accepts_matching_outputs() {
        let history = vec![
            HistoryItem::AssistantToolCalls {
                text: None,
                reasoning_content: None,
                reasoning_wire: None,
                calls: vec![tool_call("historical-call")],
            },
            HistoryItem::assistant("legacy continuation"),
            HistoryItem::AssistantToolCalls {
                text: Some("tail working".into()),
                reasoning_content: None,
                reasoning_wire: None,
                calls: vec![tool_call("tail-call")],
            },
        ];
        let frames = history_items_to_frames(&history)
            .into_iter()
            .enumerate()
            .map(|(index, mut frame)| {
                frame.runtime_frame_id = Some(RuntimeFrameId::from_persisted(index as u64 + 1));
                frame
            })
            .collect::<Vec<_>>();
        let mut incremental = ProtocolAppendState::from_frames(&frames, Some(2))
            .expect("legacy incomplete group remains representable");

        assert!(incremental.has_incomplete_tool_call_groups());
        let before = incremental.clone();
        let error = incremental
            .append(
                history.len(),
                RuntimeFrameId::from_persisted(4),
                &ProtocolFrameItem::assistant("must be rejected"),
                Some(2),
            )
            .expect_err("non-ToolOutput append must remain blocked");
        assert!(error.to_string().contains("incomplete"));
        assert_eq!(incremental, before);

        let error = incremental
            .append(
                history.len(),
                RuntimeFrameId::from_persisted(4),
                &ProtocolFrameItem::ToolOutput {
                    call_id: "historical-call".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
                Some(2),
            )
            .expect_err("historical incomplete groups cannot be repaired");
        assert!(error.to_string().contains("orphan tool output"));
        assert_eq!(incremental, before);

        incremental
            .append(
                history.len(),
                RuntimeFrameId::from_persisted(4),
                &ProtocolFrameItem::ToolOutput {
                    call_id: "tail-call".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
                Some(2),
            )
            .expect("matching tail ToolOutput is accepted despite historical state");
        assert!(incremental.has_historical_incomplete_tool_call_groups());
        assert!(incremental.has_incomplete_tool_call_groups());
        assert!(incremental.incomplete_tool_call_ids().is_empty());

        let groups = incremental.tool_call_groups();
        assert_eq!(
            groups
                .iter()
                .map(|group| group.call_ids[0].as_str())
                .collect::<Vec<_>>(),
            vec!["historical-call", "tail-call"]
        );

        let error = incremental
            .append(
                history.len() + 1,
                RuntimeFrameId::from_persisted(5),
                &ProtocolFrameItem::assistant("still rejected"),
                Some(2),
            )
            .expect_err("historical incomplete state remains globally blocking");
        assert!(error.to_string().contains("incomplete"));
    }

    #[test]
    fn validation_rejects_orphan_tool_output() {
        let history = vec![
            HistoryItem::user("question"),
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
                images: Vec::new(),
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
                reasoning_wire: None,
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
                reasoning_wire: None,
                calls: vec![tool_call("call-1")],
            },
        ];

        let error = validate_history_items_complete(&history, Some(1))
            .expect_err("dangling assistant tool calls must fail");
        assert!(error.to_string().contains("dangling assistant tool calls"));
    }
}
