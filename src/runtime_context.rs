#![allow(dead_code)]

use crate::context_tree::{ContextNodeId, ContextTreeState};
use crate::context_view::ContextViewProjection;
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
use crate::request_builder::HistoryItem;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

fn is_zero(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct RuntimeFrameId(u64);

impl RuntimeFrameId {
    pub(crate) fn from_seed(seed: &RuntimeFrameIdSeed<'_>) -> Self {
        Self(stable_fnv1a_64(&seed.canonical_key()))
    }

    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }

    pub(crate) fn from_persisted(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeFrameIdSeed<'a> {
    pub frame_kind: RuntimeFrameKind,
    pub source: RuntimeSource,
    pub ordinal: u32,
    pub stable_key: &'a str,
    pub source_span: Option<SourceSpan>,
}

impl RuntimeFrameIdSeed<'_> {
    fn canonical_key(&self) -> String {
        let span = self
            .source_span
            .map(|span| format!("{}-{}", span.start_sequence, span.end_sequence))
            .unwrap_or_else(|| "none".to_string());
        format!(
            "runtime-frame:v1:{:?}:{:?}:{}:{}:{}",
            self.frame_kind, self.source, self.ordinal, self.stable_key, span
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SourceSpan {
    pub start_sequence: u64,
    pub end_sequence: u64,
}

impl SourceSpan {
    pub(crate) fn new(start_sequence: u64, end_sequence: u64) -> Result<Self> {
        ensure!(
            start_sequence <= end_sequence,
            "source span start_sequence must be <= end_sequence"
        );
        Ok(Self {
            start_sequence,
            end_sequence,
        })
    }

    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.start_sequence <= other.end_sequence && other.start_sequence <= self.end_sequence
    }

    pub(crate) fn contains(self, other: Self) -> bool {
        self.start_sequence <= other.start_sequence && other.end_sequence <= self.end_sequence
    }

    pub(crate) fn covered_by_any(self, spans: &[Self]) -> bool {
        spans.iter().copied().any(|span| span.contains(self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimeFrameKind {
    System,
    Developer,
    User,
    Assistant,
    ToolCall,
    ToolOutput,
    Reasoning,
    ContextBlock,
    Summary,
    FoldedOutput,
    PromptContributor,
    Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimeSource {
    Transcript,
    ContextView,
    ContextTree,
    FoldedOutput,
    SummaryArtifact,
    PromptContributor,
    SessionState,
    Derived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FrameVisibility {
    Active,
    Folded,
    Retired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimePromptRole {
    System,
    Developer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimePromptPayload {
    pub role: RuntimePromptRole,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimeFrameProvenance {
    pub source: RuntimeSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_span: Option<SourceSpan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

impl RuntimeFrameProvenance {
    pub(crate) fn new(source: RuntimeSource) -> Self {
        Self {
            source,
            label: None,
            source_span: None,
            source_id: None,
        }
    }

    pub(crate) fn with_span(mut self, source_span: SourceSpan) -> Self {
        self.source_span = Some(source_span);
        self
    }

    pub(crate) fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimeFrame {
    pub id: RuntimeFrameId,
    pub kind: RuntimeFrameKind,
    pub visibility: FrameVisibility,
    pub provenance: RuntimeFrameProvenance,
    /// Exact request-bearing data.  Summaries are display metadata and must never
    /// be used to reconstruct or validate protocol state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<ProtocolFrameItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Exact detached prompt material. Display summaries must never substitute it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_payload: Option<RuntimePromptPayload>,
}

impl RuntimeFrame {
    pub(crate) fn new(
        kind: RuntimeFrameKind,
        visibility: FrameVisibility,
        provenance: RuntimeFrameProvenance,
        seed: RuntimeFrameIdSeed<'_>,
    ) -> Self {
        Self {
            id: RuntimeFrameId::from_seed(&seed),
            kind,
            visibility,
            provenance,
            protocol: None,
            summary: None,
            prompt_payload: None,
        }
    }

    pub(crate) fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub(crate) fn with_protocol(mut self, protocol: ProtocolFrameItem) -> Self {
        self.protocol = Some(protocol);
        self
    }

    pub(crate) fn with_prompt_payload(mut self, prompt_payload: RuntimePromptPayload) -> Self {
        self.prompt_payload = Some(prompt_payload);
        self
    }

    pub(crate) fn durable_identity_key(&self) -> String {
        let span = self
            .provenance
            .source_span
            .map(|span| format!("{}-{}", span.start_sequence, span.end_sequence))
            .unwrap_or_else(|| "none".into());
        // Bindings survive compaction and pruning, so their key may only use
        // lineage identity. In particular, protocol bodies and display
        // summaries are mutable representations, not identity.
        let protocol_identity = match self.protocol.as_ref() {
            Some(ProtocolFrameItem::ToolOutput { call_id, .. }) => {
                format!("tool-call:{call_id}")
            }
            _ => "none".into(),
        };
        format!(
            "runtime-frame-binding:v2:{:?}:{:?}:{}:{}:{}",
            self.kind,
            self.provenance.source,
            self.provenance.source_id.as_deref().unwrap_or("none"),
            span,
            protocol_identity,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ActiveContextMetadata {
    pub branch_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_branch_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_detail_block_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visible_block_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pinned_block_ids: Vec<String>,
}

impl ActiveContextMetadata {
    pub(crate) fn new(branch_id: impl Into<String>) -> Self {
        Self {
            branch_id: branch_id.into(),
            parent_branch_id: None,
            active_node_id: None,
            open_detail_block_id: None,
            visible_block_ids: Vec::new(),
            pinned_block_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FoldedOutputReference {
    pub output_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_span: Option<SourceSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CompactionState {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retired_source_spans: Vec<SourceSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compacted_frame_ids: Vec<RuntimeFrameId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_frame_ids: Vec<RuntimeFrameId>,
    /// Protections requested independently of the live protocol turn.  Keeping
    /// these separate lets turn protection be recomputed as turns complete.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_protected_frame_ids: Vec<RuntimeFrameId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turn_protected_frame_ids: Vec<RuntimeFrameId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptContributorKind {
    SystemPrelude,
    DeveloperPrelude,
    SkillMaterial,
    RuntimeContext,
    ContextMaterial,
    ContextIndex,
    TranscriptFrame,
    Evidence,
    FoldedOutputSummary,
    CurrentTurn,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptContributorPlaceholder {
    pub contributor_id: String,
    pub kind: PromptContributorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub provenance: RuntimeFrameProvenance,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frame_ids: Vec<RuntimeFrameId>,
    /// Source protocol representation anchors. These only suppress detached
    /// fallback while selected and are deliberately not retention authority.
    ///
    /// In contrast, `frame_ids` are retention authority: every referenced
    /// frame participates in compaction protection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_frame_ids: Vec<RuntimeFrameId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimeChildSession {
    pub parent_run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: String,
    pub summary: String,
    pub timestamp_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuntimeSnapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_sequence: Option<u64>,
    /// Durable identity of the checkout which established this cursor.  Older
    /// transcripts predate checkout identity and therefore belong to revision 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub context_scope_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_turn_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_segment_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frames: Vec<RuntimeFrame>,
    pub context_tree: ContextTreeState,
    pub context_view: ContextViewProjection,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRecord>,
    pub active_context: ActiveContextMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub folded_outputs: Vec<FoldedOutputReference>,
    pub compaction: CompactionState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_contributors: Vec<PromptContributorPlaceholder>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_sessions: Vec<RuntimeChildSession>,
}

impl RuntimeSnapshot {
    pub(crate) fn new(branch_id: impl Into<String>) -> Self {
        Self {
            session_id: None,
            latest_model: None,
            leaf_sequence: None,
            context_scope_revision: 0,
            current_turn_id: None,
            current_segment_id: None,
            frames: Vec::new(),
            context_tree: ContextTreeState::with_default_root(),
            context_view: ContextViewProjection::default(),
            evidence: Vec::new(),
            active_context: ActiveContextMetadata::new(branch_id),
            folded_outputs: Vec::new(),
            compaction: CompactionState {
                retired_source_spans: Vec::new(),
                compacted_frame_ids: Vec::new(),
                protected_frame_ids: Vec::new(),
                explicit_protected_frame_ids: Vec::new(),
                turn_protected_frame_ids: Vec::new(),
            },
            prompt_contributors: Vec::new(),
            child_sessions: Vec::new(),
        }
    }

    pub(crate) fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub(crate) fn with_latest_model(mut self, latest_model: impl Into<String>) -> Self {
        self.latest_model = Some(latest_model.into());
        self
    }

    pub(crate) fn with_leaf_sequence(mut self, leaf_sequence: u64) -> Self {
        self.leaf_sequence = Some(leaf_sequence);
        self
    }

    pub(crate) fn with_context_scope_revision(mut self, revision: u64) -> Self {
        self.context_scope_revision = revision;
        self
    }

    pub(crate) fn with_current_turn_id(mut self, current_turn_id: u64) -> Self {
        self.current_turn_id = Some(current_turn_id);
        self
    }

    pub(crate) fn with_current_segment_id(mut self, current_segment_id: u64) -> Self {
        self.current_segment_id = Some(current_segment_id);
        self
    }

    pub(crate) fn push_frame(&mut self, frame: RuntimeFrame) {
        self.frames.push(frame);
    }

    pub(crate) fn set_context_tree(&mut self, context_tree: ContextTreeState) {
        self.context_tree = context_tree;
    }

    pub(crate) fn set_context_view(&mut self, context_view: ContextViewProjection) {
        self.context_view = context_view;
    }

    pub(crate) fn overlaps_retired_source_span(&self, span: SourceSpan) -> bool {
        self.compaction
            .retired_source_spans
            .iter()
            .copied()
            .any(|retired| retired.overlaps(span))
    }

    pub(crate) fn set_evidence(&mut self, evidence: Vec<EvidenceRecord>) {
        self.evidence = evidence;
    }

    pub(crate) fn push_folded_output(&mut self, folded_output: FoldedOutputReference) {
        self.folded_outputs.push(folded_output);
    }

    pub(crate) fn push_prompt_contributor(
        &mut self,
        prompt_contributor: PromptContributorPlaceholder,
    ) {
        self.prompt_contributors.push(prompt_contributor);
    }

    pub(crate) fn push_child_session(&mut self, child_session: RuntimeChildSession) {
        self.child_sessions.push(child_session);
    }

    pub(crate) fn set_protected_frame_ids(&mut self, frame_ids: Vec<RuntimeFrameId>) {
        self.compaction.explicit_protected_frame_ids = frame_ids;
        self.recompute_protected_frame_ids();
    }

    pub(crate) fn set_turn_protected_frame_ids(&mut self, frame_ids: Vec<RuntimeFrameId>) {
        self.compaction.turn_protected_frame_ids = frame_ids;
        self.recompute_protected_frame_ids();
    }

    pub(crate) fn recompute_protected_frame_ids(&mut self) {
        let mut protected = self.compaction.explicit_protected_frame_ids.clone();
        protected.extend(self.compaction.turn_protected_frame_ids.iter().copied());
        protected.extend(
            self.prompt_contributors
                .iter()
                .flat_map(|contributor| contributor.frame_ids.iter().copied()),
        );
        protected.sort();
        protected.dedup();
        self.compaction.protected_frame_ids = protected;
    }

    /// Returns all source material retained by prompt contributors. Frame
    /// references are authoritative even when the contributor itself has no
    /// span or the referenced frame is folded or retired.
    pub(crate) fn prompt_contributor_source_spans(&self) -> Result<Vec<SourceSpan>> {
        let frames = self
            .frames
            .iter()
            .map(|frame| (frame.id, frame))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut spans = Vec::new();
        for contributor in &self.prompt_contributors {
            if let Some(span) = contributor.provenance.source_span {
                spans.push(span);
            }
            for frame_id in &contributor.frame_ids {
                let frame = frames.get(frame_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "prompt contributor '{}' references unknown runtime frame {}",
                        contributor.contributor_id,
                        frame_id.as_u64()
                    )
                })?;
                if let Some(span) = frame.provenance.source_span {
                    spans.push(span);
                }
            }
        }
        Ok(spans)
    }

    pub(crate) fn active_prompt_payload_contributors(
        &self,
    ) -> Vec<(&PromptContributorPlaceholder, &RuntimeFrame)> {
        self.prompt_contributors
            .iter()
            .filter_map(|contributor| {
                contributor
                    .frame_ids
                    .iter()
                    .find_map(|id| {
                        self.frames.iter().find(|frame| {
                            frame.id == *id
                                && frame.visibility == FrameVisibility::Active
                                && frame.prompt_payload.is_some()
                        })
                    })
                    .map(|frame| (contributor, frame))
            })
            .collect()
    }

    /// Ordered active protocol authority. Metadata may appear anywhere in
    /// `frames`; callers must use this projection rather than storage indexes.
    pub(crate) fn active_protocol_frames(&self) -> Vec<ProtocolFrame> {
        self.frames
            .iter()
            .filter(|frame| frame.visibility == FrameVisibility::Active)
            .filter_map(|frame| {
                frame.protocol.clone().map(|item| ProtocolFrame {
                    runtime_frame_id: Some(frame.id),
                    source_provenance: Some(frame.provenance.clone()),
                    history_index: 0,
                    item,
                })
            })
            .enumerate()
            .map(|(history_index, mut frame)| {
                frame.history_index = history_index;
                frame
            })
            .collect()
    }

    pub(crate) fn active_history_items(&self) -> Vec<HistoryItem> {
        self.active_protocol_frames()
            .iter()
            .map(ProtocolFrame::to_history_item)
            .collect()
    }

    pub(crate) fn validate_references(&self) -> Result<()> {
        let ids = self
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            ids.len() == self.frames.len(),
            "runtime snapshot contains duplicate frame id"
        );
        let contributor_ids = self
            .prompt_contributors
            .iter()
            .map(|contributor| &contributor.contributor_id)
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            contributor_ids.len() == self.prompt_contributors.len(),
            "runtime snapshot contains duplicate prompt contributor id"
        );
        for id in self
            .compaction
            .protected_frame_ids
            .iter()
            .chain(self.compaction.explicit_protected_frame_ids.iter())
            .chain(self.compaction.turn_protected_frame_ids.iter())
            .chain(self.compaction.compacted_frame_ids.iter())
            .chain(self.prompt_contributors.iter().flat_map(|contributor| {
                contributor
                    .frame_ids
                    .iter()
                    .chain(contributor.source_frame_ids.iter())
            }))
        {
            ensure!(
                ids.contains(id),
                "runtime snapshot has dangling frame reference"
            );
        }
        Ok(())
    }
}

/// The sole context payload accepted by the TUI.  It deliberately shares the
/// projection rather than copying output bodies into an event-specific model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeActiveContext {
    pub session_id: String,
    pub leaf_sequence: u64,
    pub context_scope_revision: u64,
    pub active_context: ActiveContextMetadata,
    pub context_tree: ContextTreeState,
    pub context_view: ContextViewProjection,
}

impl TryFrom<&RuntimeSnapshot> for RuntimeActiveContext {
    type Error = anyhow::Error;

    fn try_from(snapshot: &RuntimeSnapshot) -> Result<Self> {
        let session_id = snapshot
            .session_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("runtime context is missing session_id"))?;
        let leaf_sequence = snapshot
            .leaf_sequence
            .ok_or_else(|| anyhow::anyhow!("runtime context is missing leaf_sequence"))?;
        let active_node = snapshot
            .active_context
            .active_node_id
            .as_ref()
            .map(|id| {
                ContextNodeId::new(id.clone())
                    .map_err(|_| anyhow::anyhow!("invalid active node id '{id}'"))
            })
            .transpose()?;
        ensure!(
            active_node.as_ref() == snapshot.context_tree.active_node_id(),
            "runtime active node does not match context tree"
        );
        ensure!(
            snapshot.active_context.open_detail_block_id
                == snapshot.context_view.provider_open_detail_block_id(),
            "runtime open detail does not match provider context view"
        );
        ensure!(
            snapshot.active_context.visible_block_ids
                == snapshot.context_view.provider_visible_block_ids(),
            "runtime visible blocks do not match provider context view"
        );
        ensure!(
            snapshot.active_context.pinned_block_ids
                == snapshot.context_view.provider_pinned_block_ids(),
            "runtime pinned blocks do not match provider context view"
        );
        for output in snapshot.context_view.provider_folded_outputs() {
            let reference = snapshot
                .folded_outputs
                .iter()
                .find(|reference| {
                    reference.output_id == output.output_id
                        && reference.node_id == output.node_id
                        && reference.call_id == output.call_id
                        && reference.tool_name == output.tool_name
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "provider folded output '{}' lacks matching runtime reference",
                        output.output_id
                    )
                })?;
            let expected_span = match (output.source_start_sequence, output.source_end_sequence) {
                (Some(start), Some(end)) => Some(SourceSpan::new(start, end)?),
                (Some(start), None) => Some(SourceSpan::new(start, start)?),
                _ => None,
            };
            ensure!(
                reference.source_span == expected_span,
                "provider folded output '{}' has mismatched source span",
                output.output_id
            );
        }
        for reference in &snapshot.folded_outputs {
            if snapshot
                .context_view
                .is_active_folded_output(&reference.output_id)
            {
                ensure!(
                    !snapshot
                        .context_view
                        .is_compacted_folded_output(&reference.output_id),
                    "compacted folded output '{}' is active",
                    reference.output_id
                );
            }
        }
        Ok(Self {
            session_id,
            leaf_sequence,
            context_scope_revision: snapshot.context_scope_revision,
            active_context: snapshot.active_context.clone(),
            context_tree: snapshot.context_tree.clone(),
            context_view: snapshot.context_view.clone(),
        })
    }
}

fn stable_fnv1a_64(input: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_ids_are_deterministic_for_identical_seed() {
        let span = SourceSpan::new(11, 17).expect("valid source span");
        let seed = RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Assistant,
            source: RuntimeSource::Transcript,
            ordinal: 3,
            stable_key: "assistant-message",
            source_span: Some(span),
        };

        let first = RuntimeFrameId::from_seed(&seed);
        let second = RuntimeFrameId::from_seed(&seed);

        assert_eq!(first, second);
    }

    #[test]
    fn frame_ids_change_when_seed_changes() {
        let left = RuntimeFrameId::from_seed(&RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Assistant,
            source: RuntimeSource::Transcript,
            ordinal: 1,
            stable_key: "assistant-message",
            source_span: Some(SourceSpan::new(11, 17).expect("valid source span")),
        });
        let right = RuntimeFrameId::from_seed(&RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Assistant,
            source: RuntimeSource::Transcript,
            ordinal: 2,
            stable_key: "assistant-message",
            source_span: Some(SourceSpan::new(11, 17).expect("valid source span")),
        });

        assert_ne!(left, right);
    }

    #[test]
    fn runtime_snapshot_constructor_initializes_empty_scaffold() {
        let snapshot = RuntimeSnapshot::new("main")
            .with_session_id("session-1")
            .with_latest_model("gpt-5")
            .with_leaf_sequence(9)
            .with_current_turn_id(42);

        assert_eq!(snapshot.session_id.as_deref(), Some("session-1"));
        assert_eq!(snapshot.latest_model.as_deref(), Some("gpt-5"));
        assert_eq!(snapshot.leaf_sequence, Some(9));
        assert_eq!(snapshot.current_turn_id, Some(42));
        assert_eq!(snapshot.active_context.branch_id, "main");
        assert!(snapshot.frames.is_empty());
        assert_eq!(snapshot.context_tree.root_node_id().as_str(), "root");
        assert!(snapshot.context_view.blocks.is_empty());
        assert!(snapshot.evidence.is_empty());
        assert!(snapshot.folded_outputs.is_empty());
        assert!(snapshot.prompt_contributors.is_empty());
        assert!(snapshot.child_sessions.is_empty());
        assert!(snapshot.compaction.retired_source_spans.is_empty());
    }

    #[test]
    fn source_span_rejects_inverted_ranges() {
        let error = SourceSpan::new(9, 3).expect_err("inverted spans must fail");

        assert!(
            error
                .to_string()
                .contains("source span start_sequence must be <= end_sequence")
        );
    }

    #[test]
    fn protocol_projection_keeps_frame_ids_across_interspersed_metadata() {
        let mut snapshot = RuntimeSnapshot::new("main");
        let user = RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Derived,
                ordinal: 0,
                stable_key: "first",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: crate::user_content::UserMessageContent::new("same", Vec::new()),
        });
        let second = RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Derived,
                ordinal: 1,
                stable_key: "second",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: crate::user_content::UserMessageContent::new("same", Vec::new()),
        });
        let first_id = user.id;
        let second_id = second.id;
        snapshot.push_frame(user);
        snapshot.push_frame(RuntimeFrame::new(
            RuntimeFrameKind::Metadata,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::Metadata,
                source: RuntimeSource::Derived,
                ordinal: 0,
                stable_key: "metadata",
                source_span: None,
            },
        ));
        snapshot.push_frame(second);

        let projected = snapshot.active_protocol_frames();
        assert_eq!(
            projected
                .iter()
                .map(|frame| frame.runtime_frame_id)
                .collect::<Vec<_>>(),
            vec![Some(first_id), Some(second_id)]
        );
        assert_eq!(
            snapshot.active_history_items(),
            vec![HistoryItem::user("same"), HistoryItem::user("same")]
        );
    }

    #[test]
    fn snapshot_reference_validation_rejects_duplicate_ids() {
        let mut snapshot = RuntimeSnapshot::new("main");
        let frame = RuntimeFrame::new(
            RuntimeFrameKind::Metadata,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::Metadata,
                source: RuntimeSource::Derived,
                ordinal: 0,
                stable_key: "one",
                source_span: None,
            },
        );
        snapshot.push_frame(frame.clone());
        snapshot.push_frame(frame);
        assert!(snapshot.validate_references().is_err());
    }
}
