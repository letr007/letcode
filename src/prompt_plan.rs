#![allow(dead_code)]

use crate::config::ApiProtocol;
use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
use crate::request_builder::{
    HistoryItem, HistoryToolCall, PromptMessage, PromptMessageOrigin, PromptRole,
    estimate_history_item_tokens,
};
use crate::runtime_context::{
    PromptContributorKind, RuntimeFrameId, RuntimeFrameProvenance, RuntimePromptRole,
    RuntimeSnapshot, RuntimeSource,
};
use crate::user_content::UserMessageContent;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptSegmentRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptSegmentStability {
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptSegmentRetention {
    Required,
    Retained,
    Droppable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct PromptSegmentProtection {
    pub current_turn: bool,
    pub protocol_boundary: bool,
    pub retained: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptTokenEstimate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptCacheBoundaryKind {
    StablePrefixEnd,
    VolatileRegionStart,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptCacheMetadata {
    pub cache_eligible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary: Option<PromptCacheBoundaryKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptSegmentSource {
    pub order: u32,
    pub contributor_kind: PromptContributorKind,
    pub provenance: RuntimeFrameProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptContributor {
    pub id: String,
    pub kind: PromptContributorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub order: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segment_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PromptSegment {
    pub id: String,
    pub order: u32,
    pub role: PromptSegmentRole,
    pub contributor_id: String,
    pub source: PromptSegmentSource,
    pub stability: PromptSegmentStability,
    pub retention: PromptSegmentRetention,
    pub protection: PromptSegmentProtection,
    pub cache: PromptCacheMetadata,
    pub tokens: PromptTokenEstimate,
    pub text: String,
    pub content: PromptSegmentContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum PromptSegmentContent {
    Text {
        text: String,
    },
    UserContent {
        content: UserMessageContent,
    },
    AssistantToolCalls {
        text: Option<String>,
        calls: Vec<HistoryToolCall>,
    },
    ToolOutput {
        call_id: String,
        output_json: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptPlan {
    pub protocol: ApiProtocol,
    pub model_id: String,
    pub contributors: Vec<PromptContributor>,
    pub segments: Vec<PromptSegment>,
    pub stable_prefix_end: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PromptPlanTokenReport {
    pub total_prompt_tokens: u64,
    pub stable_prompt_tokens: u64,
    pub volatile_prompt_tokens: u64,
    pub cacheable_prefix_tokens: u64,
    pub stable_after_boundary_tokens: u64,
    pub first_volatile_index: Option<usize>,
}

impl PromptPlan {
    pub(crate) fn estimated_input_tokens(&self) -> u64 {
        self.segments
            .iter()
            .map(|segment| segment.tokens.estimated_input_tokens.unwrap_or(0))
            .sum()
    }

    pub(crate) fn cacheable_prefix_len(&self) -> usize {
        self.segments
            .iter()
            .take_while(|segment| segment.stability == PromptSegmentStability::Stable)
            .count()
    }

    pub(crate) fn token_report(&self) -> PromptPlanTokenReport {
        let mut report = PromptPlanTokenReport::default();
        let prefix_len = self.cacheable_prefix_len();
        report.first_volatile_index = self
            .segments
            .iter()
            .position(|segment| segment.stability == PromptSegmentStability::Volatile);
        for (index, segment) in self.segments.iter().enumerate() {
            let tokens = segment.tokens.estimated_input_tokens.unwrap_or(0);
            report.total_prompt_tokens += tokens;
            if segment.stability == PromptSegmentStability::Stable {
                report.stable_prompt_tokens += tokens;
            } else {
                report.volatile_prompt_tokens += tokens;
            }
            if index < prefix_len {
                report.cacheable_prefix_tokens += tokens;
            }
        }
        report.stable_after_boundary_tokens = report
            .stable_prompt_tokens
            .saturating_sub(report.cacheable_prefix_tokens);
        report
    }

    pub(crate) fn stable_prefix_hash(&self) -> Option<&str> {
        self.segments.iter().find_map(|segment| {
            (segment.cache.boundary == Some(PromptCacheBoundaryKind::StablePrefixEnd))
                .then(|| segment.cache.prefix_hash.as_deref())
                .flatten()
        })
    }

    pub(crate) fn segment(&self, id: &str) -> Option<&PromptSegment> {
        self.segments.iter().find(|segment| segment.id == id)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PromptPlanBuildInput<'a> {
    pub protocol: ApiProtocol,
    pub model_id: &'a str,
    pub prelude: &'a [PromptMessage],
    pub snapshot: &'a RuntimeSnapshot,
    pub selected_frames: &'a [ProtocolFrame],
    pub protected_suffix_len: usize,
    pub evidence_message: Option<&'a str>,
    pub selected_evidence_ids: &'a [String],
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedRuntimePromptMaterial {
    pub contributor_id: String,
    pub contributor_kind: PromptContributorKind,
    pub label: Option<String>,
    pub provenance: RuntimeFrameProvenance,
    pub frame_id: RuntimeFrameId,
    pub role: RuntimePromptRole,
    pub text: String,
}

pub(crate) fn build_prompt_plan(input: PromptPlanBuildInput<'_>) -> PromptPlan {
    let mut builder = PromptPlanBuilder::new(input.protocol, input.model_id);

    for message in input.prelude {
        let classification = classify_prelude_message(message);
        builder.push_segment(NewPromptSegment {
            contributor_kind: classification.kind,
            contributor_label: classification.label,
            role: classification.role,
            stability: classification.stability,
            retention: PromptSegmentRetention::Required,
            protection: PromptSegmentProtection {
                retained: true,
                ..PromptSegmentProtection::default()
            },
            provenance: RuntimeFrameProvenance::new(classification.source),
            source_key: Some(message_stable_key(message)),
            source_label: Some("prelude".to_string()),
            token_estimate: estimate_prompt_message_tokens(message),
            text: message.text.clone(),
            content: PromptSegmentContent::Text {
                text: message.text.clone(),
            },
        });
    }

    let selected_ids = input
        .selected_frames
        .iter()
        .filter_map(|frame| frame.runtime_frame_id)
        .collect::<std::collections::BTreeSet<_>>();
    let prompt_material = input
        .snapshot
        .active_prompt_payload_contributors()
        .into_iter()
        .filter_map(|(contributor, frame)| {
            let payload = frame.prompt_payload.as_ref()?;
            (!contributor
                .source_frame_ids
                .iter()
                .any(|id| selected_ids.contains(id)))
            .then(|| SelectedRuntimePromptMaterial {
                contributor_id: contributor.contributor_id.clone(),
                contributor_kind: contributor.kind,
                label: contributor.label.clone(),
                provenance: frame.provenance.clone(),
                frame_id: frame.id,
                role: payload.role,
                text: payload.text.clone(),
            })
        })
        .collect::<Vec<_>>();
    for material in &prompt_material {
        let role = match material.role {
            RuntimePromptRole::System => PromptSegmentRole::System,
            RuntimePromptRole::Developer => PromptSegmentRole::Developer,
        };
        builder.push_segment(NewPromptSegment {
            contributor_kind: material.contributor_kind,
            contributor_label: material.label.clone(),
            role,
            stability: PromptSegmentStability::Stable,
            retention: PromptSegmentRetention::Required,
            protection: PromptSegmentProtection {
                retained: true,
                ..PromptSegmentProtection::default()
            },
            provenance: material.provenance.clone(),
            source_key: Some(format!(
                "{}:{}",
                material.contributor_id,
                material.frame_id.as_u64()
            )),
            source_label: Some(material.contributor_id.clone()),
            token_estimate: Some(estimate_history_item_tokens(&HistoryItem::ContextSummary {
                text: material.text.clone(),
            })),
            text: material.text.clone(),
            content: PromptSegmentContent::Text {
                text: material.text.clone(),
            },
        });
    }

    let protected_suffix_len = input.protected_suffix_len.min(input.selected_frames.len());
    let current_turn_start = input
        .selected_frames
        .len()
        .saturating_sub(protected_suffix_len);

    let evidence_insert_index = input
        .evidence_message
        .and_then(|_| last_user_frame_index(input.selected_frames));

    for (index, frame) in input.selected_frames.iter().enumerate() {
        if evidence_insert_index == Some(index) {
            push_evidence_segment(&mut builder, evidence_message_segment(&input));
        }
        let provenance = frame.source_provenance.clone().or_else(|| {
            frame.runtime_frame_id.and_then(|id| {
                input
                    .snapshot
                    .frames
                    .iter()
                    .find(|candidate| candidate.id == id)
                    .map(|runtime_frame| runtime_frame.provenance.clone())
            })
        });
        let mut classification =
            classify_history_frame(&frame, index >= current_turn_start, provenance.as_ref());
        if let Some(contributor) = frame.runtime_frame_id.and_then(|id| {
            input
                .snapshot
                .prompt_contributors
                .iter()
                .find(|contributor| contributor.source_frame_ids.contains(&id))
        }) {
            classification.kind = PromptContributorKind::SkillMaterial;
            classification.label = contributor.label.clone();
        }
        let provenance =
            provenance.unwrap_or_else(|| RuntimeFrameProvenance::new(classification.source));
        builder.push_segment(NewPromptSegment {
            contributor_kind: classification.kind,
            contributor_label: classification.label,
            role: classification.role,
            stability: classification.stability,
            retention: classification.retention,
            protection: PromptSegmentProtection {
                current_turn: index >= current_turn_start,
                protocol_boundary: matches!(
                    frame.item,
                    ProtocolFrameItem::AssistantToolCalls { .. }
                        | ProtocolFrameItem::ToolOutput { .. }
                ),
                retained: true,
            },
            provenance,
            source_key: Some(
                frame
                    .runtime_frame_id
                    .map(|id| id.as_u64().to_string())
                    .unwrap_or_else(|| frame.stable_prompt_key()),
            ),
            source_label: Some(frame.prompt_source_label().to_string()),
            token_estimate: Some(estimate_history_item_tokens(&frame.to_history_item())),
            text: history_item_text(&frame.to_history_item()),
            content: history_item_content(&frame.to_history_item()),
        });
    }

    if input.evidence_message.is_some() && evidence_insert_index.is_none() {
        push_evidence_segment(&mut builder, evidence_message_segment(&input));
    }

    builder.finish()
}

fn last_user_frame_index(frames: &[ProtocolFrame]) -> Option<usize> {
    frames
        .iter()
        .rposition(|frame| matches!(frame.item, ProtocolFrameItem::UserMessage { .. }))
}

struct PromptPlanBuilder {
    protocol: ApiProtocol,
    model_id: String,
    contributors: Vec<PromptContributor>,
    segments: Vec<PromptSegment>,
}

impl PromptPlanBuilder {
    fn new(protocol: ApiProtocol, model_id: &str) -> Self {
        Self {
            protocol,
            model_id: model_id.to_string(),
            contributors: Vec::new(),
            segments: Vec::new(),
        }
    }

    fn ensure_contributor(
        &mut self,
        kind: PromptContributorKind,
        label: Option<String>,
        provenance: RuntimeFrameProvenance,
    ) -> String {
        let source_id = provenance.source_id.clone().unwrap_or_default();
        let seed = format!(
            "contributor:{kind:?}:{}:{}",
            label.clone().unwrap_or_default(),
            source_id
        );
        let id = stable_hash_input(&seed);
        if !self
            .contributors
            .iter()
            .any(|contributor| contributor.id == id)
        {
            self.contributors.push(PromptContributor {
                id: id.clone(),
                kind,
                label,
                order: self.contributors.len() as u32,
                segment_ids: Vec::new(),
            });
        }
        id
    }

    fn push_segment(&mut self, new_segment: NewPromptSegment) {
        let contributor_id = self.ensure_contributor(
            new_segment.contributor_kind,
            new_segment.contributor_label,
            new_segment.provenance.clone(),
        );
        let order = self.segments.len() as u32;
        let segment_id = stable_hash_input(&format!(
            "segment:{}:{:?}:{:?}:{}:{}",
            order,
            new_segment.role,
            new_segment.contributor_kind,
            new_segment.source_key.clone().unwrap_or_default(),
            new_segment.text
        ));
        self.segments.push(PromptSegment {
            id: segment_id.clone(),
            order,
            role: new_segment.role,
            contributor_id: contributor_id.clone(),
            source: PromptSegmentSource {
                order,
                contributor_kind: new_segment.contributor_kind,
                provenance: new_segment.provenance,
                source_key: new_segment.source_key,
                source_label: new_segment.source_label,
            },
            stability: new_segment.stability,
            retention: new_segment.retention,
            protection: new_segment.protection,
            cache: PromptCacheMetadata {
                cache_eligible: false,
                boundary: None,
                prefix_hash: None,
            },
            tokens: PromptTokenEstimate {
                estimated_input_tokens: new_segment.token_estimate,
                budget_input_tokens: new_segment.token_estimate,
                actual_input_tokens: None,
            },
            text: new_segment.text,
            content: new_segment.content,
        });
        if let Some(contributor) = self
            .contributors
            .iter_mut()
            .find(|contributor| contributor.id == contributor_id)
        {
            contributor.segment_ids.push(segment_id);
        }
    }

    fn finish(mut self) -> PromptPlan {
        let stable_prefix_len = self
            .segments
            .iter()
            .take_while(|segment| segment.stability == PromptSegmentStability::Stable)
            .count();
        let stable_prefix_end = match stable_prefix_len {
            0 => None,
            len if len == self.segments.len() => len.checked_sub(1),
            len => len.checked_sub(1),
        };
        if let Some(stable_end) = stable_prefix_end {
            let prefix_hash = stable_hash_input(
                &self.segments[..=stable_end]
                    .iter()
                    .map(|segment| format!("{}:{}", segment.id, segment.text))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            if let Some(segment) = self.segments.get_mut(stable_end) {
                segment.cache.boundary = Some(PromptCacheBoundaryKind::StablePrefixEnd);
                segment.cache.prefix_hash = Some(prefix_hash.clone());
            }
            if let Some(segment) = self.segments.get_mut(stable_end + 1) {
                segment.cache.boundary = Some(PromptCacheBoundaryKind::VolatileRegionStart);
                segment.cache.prefix_hash = Some(prefix_hash);
            }
        }
        for segment in self.segments.iter_mut().take(stable_prefix_len) {
            segment.cache.cache_eligible = true;
        }
        PromptPlan {
            protocol: self.protocol,
            model_id: self.model_id,
            contributors: self.contributors,
            segments: self.segments,
            stable_prefix_end,
        }
    }
}

struct NewPromptSegment {
    contributor_kind: PromptContributorKind,
    contributor_label: Option<String>,
    role: PromptSegmentRole,
    stability: PromptSegmentStability,
    retention: PromptSegmentRetention,
    protection: PromptSegmentProtection,
    provenance: RuntimeFrameProvenance,
    source_key: Option<String>,
    source_label: Option<String>,
    token_estimate: Option<u64>,
    text: String,
    content: PromptSegmentContent,
}

struct PreludeClassification {
    kind: PromptContributorKind,
    label: Option<String>,
    role: PromptSegmentRole,
    stability: PromptSegmentStability,
    source: RuntimeSource,
}

struct HistoryClassification {
    kind: PromptContributorKind,
    label: Option<String>,
    role: PromptSegmentRole,
    stability: PromptSegmentStability,
    retention: PromptSegmentRetention,
    source: RuntimeSource,
}

fn classify_prelude_message(message: &PromptMessage) -> PreludeClassification {
    let (kind, label, stability, source) = match message.origin {
        PromptMessageOrigin::StaticPrelude => match message.role {
            PromptRole::System => (
                PromptContributorKind::SystemPrelude,
                "system_prelude",
                PromptSegmentStability::Stable,
                RuntimeSource::PromptContributor,
            ),
            PromptRole::Developer => (
                PromptContributorKind::DeveloperPrelude,
                "developer_prelude",
                PromptSegmentStability::Stable,
                RuntimeSource::PromptContributor,
            ),
        },
        PromptMessageOrigin::SkillCatalog => (
            PromptContributorKind::SkillMaterial,
            "skill_catalog",
            PromptSegmentStability::Stable,
            RuntimeSource::PromptContributor,
        ),
        PromptMessageOrigin::RuntimeClock => (
            PromptContributorKind::RuntimeContext,
            "runtime_clock",
            PromptSegmentStability::Volatile,
            RuntimeSource::ContextView,
        ),
        PromptMessageOrigin::WorkflowTurn => (
            PromptContributorKind::CurrentTurn,
            "workflow_turn",
            PromptSegmentStability::Volatile,
            RuntimeSource::ContextView,
        ),
        PromptMessageOrigin::UnreconciledSubagentContext => (
            PromptContributorKind::RuntimeContext,
            "unreconciled_subagent_context",
            PromptSegmentStability::Volatile,
            RuntimeSource::ContextView,
        ),
        PromptMessageOrigin::RuntimeContextView => (
            PromptContributorKind::RuntimeContext,
            "runtime_context_view",
            PromptSegmentStability::Volatile,
            RuntimeSource::ContextView,
        ),
    };
    PreludeClassification {
        kind,
        label: Some(label.to_string()),
        role: match message.role {
            PromptRole::System => PromptSegmentRole::System,
            PromptRole::Developer => PromptSegmentRole::Developer,
        },
        stability,
        source,
    }
}

fn classify_history_frame(
    frame: &ProtocolFrame,
    current_turn: bool,
    provenance: Option<&RuntimeFrameProvenance>,
) -> HistoryClassification {
    let (role, default_source, retention) = match &frame.item {
        ProtocolFrameItem::ContextSummary { .. } => (
            PromptSegmentRole::Developer,
            RuntimeSource::SummaryArtifact,
            PromptSegmentRetention::Retained,
        ),
        ProtocolFrameItem::UserMessage { .. } => (
            PromptSegmentRole::User,
            RuntimeSource::Transcript,
            PromptSegmentRetention::Retained,
        ),
        ProtocolFrameItem::InternalContinuation { .. } => (
            PromptSegmentRole::User,
            RuntimeSource::Transcript,
            PromptSegmentRetention::Retained,
        ),
        ProtocolFrameItem::AssistantText { .. } => (
            PromptSegmentRole::Assistant,
            RuntimeSource::Transcript,
            PromptSegmentRetention::Retained,
        ),
        ProtocolFrameItem::AssistantToolCalls { .. } => (
            PromptSegmentRole::Assistant,
            RuntimeSource::Transcript,
            PromptSegmentRetention::Required,
        ),
        ProtocolFrameItem::ToolOutput { .. } => (
            PromptSegmentRole::Tool,
            RuntimeSource::Transcript,
            PromptSegmentRetention::Required,
        ),
    };

    if let Some(provenance) = provenance
        && is_mutable_runtime_projection(provenance.source)
    {
        return HistoryClassification {
            kind: if current_turn {
                PromptContributorKind::CurrentTurn
            } else {
                runtime_projection_contributor_kind(provenance.source)
            },
            label: Some(runtime_projection_label(provenance.source).to_string()),
            role,
            stability: PromptSegmentStability::Volatile,
            retention,
            source: provenance.source,
        };
    }

    if provenance.is_none_or(|provenance| provenance.source == RuntimeSource::Derived)
        && let ProtocolFrameItem::ContextSummary { text } = &frame.item
        && let Some((kind, label)) = classify_context_label(text)
    {
        let contributor_kind = if current_turn {
            PromptContributorKind::CurrentTurn
        } else {
            kind
        };
        return HistoryClassification {
            kind: contributor_kind,
            label: Some(label.to_string()),
            role: PromptSegmentRole::Developer,
            stability: PromptSegmentStability::Volatile,
            retention: PromptSegmentRetention::Retained,
            source: RuntimeSource::ContextView,
        };
    }

    HistoryClassification {
        kind: if current_turn {
            PromptContributorKind::CurrentTurn
        } else {
            PromptContributorKind::TranscriptFrame
        },
        label: Some(if current_turn {
            "current_turn".to_string()
        } else {
            frame.prompt_source_label().to_string()
        }),
        role,
        stability: if current_turn {
            PromptSegmentStability::Volatile
        } else {
            PromptSegmentStability::Stable
        },
        retention,
        source: provenance.map_or(default_source, |provenance| provenance.source),
    }
}

fn is_mutable_runtime_projection(source: RuntimeSource) -> bool {
    matches!(
        source,
        RuntimeSource::ContextView
            | RuntimeSource::ContextTree
            | RuntimeSource::FoldedOutput
            | RuntimeSource::SummaryArtifact
            | RuntimeSource::SessionState
    )
}

fn runtime_projection_contributor_kind(source: RuntimeSource) -> PromptContributorKind {
    match source {
        RuntimeSource::FoldedOutput => PromptContributorKind::FoldedOutputSummary,
        RuntimeSource::SummaryArtifact => PromptContributorKind::ContextMaterial,
        RuntimeSource::ContextView | RuntimeSource::ContextTree | RuntimeSource::SessionState => {
            PromptContributorKind::RuntimeContext
        }
        RuntimeSource::Transcript | RuntimeSource::PromptContributor | RuntimeSource::Derived => {
            unreachable!("only mutable runtime projections are classified here")
        }
    }
}

fn runtime_projection_label(source: RuntimeSource) -> &'static str {
    match source {
        RuntimeSource::ContextView => "runtime_context",
        RuntimeSource::ContextTree => "runtime_context_tree",
        RuntimeSource::FoldedOutput => "folded_output_summaries",
        RuntimeSource::SummaryArtifact => "context_summaries",
        RuntimeSource::SessionState => "session_state",
        RuntimeSource::Transcript | RuntimeSource::PromptContributor | RuntimeSource::Derived => {
            unreachable!("only mutable runtime projections are classified here")
        }
    }
}

fn classify_context_label(text: &str) -> Option<(PromptContributorKind, &'static str)> {
    if text.starts_with("[Context: Hard Context]") || text.starts_with("[Context: Pinned Context]")
    {
        Some((PromptContributorKind::RuntimeContext, "runtime_context"))
    } else if text.starts_with("[Context: Index]") {
        Some((PromptContributorKind::ContextIndex, "context_index"))
    } else if text.starts_with("[Context: Folded Outputs]") {
        Some((
            PromptContributorKind::FoldedOutputSummary,
            "folded_output_summaries",
        ))
    } else if text.starts_with("[Context: Active Tail]")
        || text.starts_with("[Context: Opened Details]")
    {
        Some((PromptContributorKind::CurrentTurn, "current_turn_context"))
    } else if text.starts_with("[Context: Summaries]") {
        Some((PromptContributorKind::ContextMaterial, "context_summaries"))
    } else {
        None
    }
}

fn history_item_text(item: &HistoryItem) -> String {
    match item {
        HistoryItem::ContextSummary { text } => render_context_summary(text),
        HistoryItem::UserMessage { content } => content.prompt_plan_text(),
        HistoryItem::InternalContinuation { text } => text.clone(),
        HistoryItem::AssistantText { text } => text.clone(),
        HistoryItem::AssistantToolCalls { text, calls } => {
            let calls_text = calls
                .iter()
                .map(|call| format!("{} {} {}", call.call_id, call.name, call.arguments_json))
                .collect::<Vec<_>>()
                .join("\n");
            match text {
                Some(text) if !text.is_empty() => format!("{text}\n{calls_text}"),
                _ => calls_text,
            }
        }
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => format!("{call_id}\n{output_json}"),
    }
}

fn render_context_summary(text: &str) -> String {
    format!("以下是当前会话的结构化摘要：\n\n{text}")
}

fn push_evidence_segment(builder: &mut PromptPlanBuilder, segment: Option<NewPromptSegment>) {
    if let Some(segment) = segment {
        builder.push_segment(segment);
    }
}

fn evidence_message_segment(input: &PromptPlanBuildInput<'_>) -> Option<NewPromptSegment> {
    let evidence_message = input.evidence_message?;
    let source_id = if input.selected_evidence_ids.is_empty() {
        None
    } else {
        Some(input.selected_evidence_ids.join(","))
    };
    let mut provenance = RuntimeFrameProvenance::new(RuntimeSource::Derived);
    if let Some(source_id) = source_id {
        provenance = provenance.with_source_id(source_id);
    }
    Some(NewPromptSegment {
        contributor_kind: PromptContributorKind::Evidence,
        contributor_label: Some("selected_evidence".to_string()),
        role: PromptSegmentRole::Developer,
        stability: PromptSegmentStability::Volatile,
        retention: PromptSegmentRetention::Droppable,
        protection: PromptSegmentProtection {
            retained: true,
            ..PromptSegmentProtection::default()
        },
        provenance,
        source_key: Some(stable_hash_input(&format!(
            "evidence:{}:{}",
            input.selected_evidence_ids.join(","),
            evidence_message
        ))),
        source_label: Some("evidence_message".to_string()),
        token_estimate: Some(estimate_text_tokens(evidence_message)),
        text: evidence_message.to_string(),
        content: PromptSegmentContent::Text {
            text: evidence_message.to_string(),
        },
    })
}

fn history_item_content(item: &HistoryItem) -> PromptSegmentContent {
    match item {
        HistoryItem::ContextSummary { text } => PromptSegmentContent::Text {
            text: render_context_summary(text),
        },
        HistoryItem::InternalContinuation { text } | HistoryItem::AssistantText { text } => {
            PromptSegmentContent::Text { text: text.clone() }
        }
        HistoryItem::UserMessage { content } => PromptSegmentContent::UserContent {
            content: content.clone(),
        },
        HistoryItem::AssistantToolCalls { text, calls } => {
            PromptSegmentContent::AssistantToolCalls {
                text: text.clone(),
                calls: calls.clone(),
            }
        }
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => PromptSegmentContent::ToolOutput {
            call_id: call_id.clone(),
            output_json: output_json.clone(),
        },
    }
}

fn estimate_prompt_message_tokens(message: &PromptMessage) -> Option<u64> {
    let json_len = serde_json::to_string(message).ok()?.len();
    Some(((json_len as u64 + 2) / 3).saturating_add(8))
}

fn estimate_text_tokens(text: &str) -> u64 {
    ((text.len() as u64 + 2) / 3).saturating_add(8)
}

fn message_stable_key(message: &PromptMessage) -> String {
    stable_hash_input(&format!("prelude:{:?}:{}", message.role, message.text))
}

fn stable_hash_input(input: &str) -> String {
    // Segment identities are opaque, deterministic SHA-256 values. Prefix
    // fingerprints themselves are generated from provider-rendered input.
    crate::request_builder::sha256_hex(input.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol_frames::history_items_to_frames;
    use crate::request_builder::HistoryToolCall;
    use crate::runtime_context::RuntimeSnapshot;
    use crate::user_content::{UserImageAttachment, UserMessageContent};

    #[test]
    fn prompt_plan_marks_stable_prefix_boundary() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[
                PromptMessage::system("system"),
                PromptMessage::developer("developer"),
                PromptMessage::developer_with_origin(
                    "[Context: Hard Context]\n- pin",
                    PromptMessageOrigin::RuntimeContextView,
                ),
            ],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[
                HistoryItem::assistant("older assistant"),
                HistoryItem::user("latest user"),
            ]),
            protected_suffix_len: 1,
            evidence_message: Some(
                "Relevant evidence:\n- [ev-1] file_excerpt src/main.rs — summary",
            ),
            selected_evidence_ids: &["ev-1".to_string()],
        });

        assert_eq!(plan.stable_prefix_end, Some(1));
        assert!(plan.stable_prefix_hash().is_some());
        assert_eq!(
            plan.segments[2].cache.boundary,
            Some(PromptCacheBoundaryKind::VolatileRegionStart)
        );
    }

    #[test]
    fn prompt_plan_has_no_stable_prefix_when_first_segment_is_volatile() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[PromptMessage::developer_with_origin(
                "[Context: Hard Context]\n- pin",
                PromptMessageOrigin::RuntimeContextView,
            )],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &[],
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        });

        assert_eq!(plan.stable_prefix_end, None);
        assert!(plan.stable_prefix_hash().is_none());
        assert!(
            plan.segments
                .iter()
                .all(|segment| segment.cache.boundary.is_none())
        );
        assert!(
            plan.segments
                .iter()
                .all(|segment| segment.cache.prefix_hash.is_none())
        );
    }

    #[test]
    fn prompt_plan_cache_metadata_for_empty_plan() {
        let plan = build_cache_test_plan(&[], &[], 0);
        let report = plan.token_report();

        assert_eq!(plan.cacheable_prefix_len(), 0);
        assert_eq!(plan.stable_prefix_end, None);
        assert_eq!(plan.stable_prefix_hash(), None);
        assert_eq!(report, PromptPlanTokenReport::default());
        assert!(plan.segments.is_empty());
        assert_token_report_invariants(&plan);
    }

    #[test]
    fn prompt_plan_cache_metadata_when_first_segment_is_volatile() {
        let plan = build_cache_test_plan(
            &[PromptMessage::developer_with_origin(
                "clock",
                PromptMessageOrigin::RuntimeClock,
            )],
            &[],
            0,
        );
        let report = plan.token_report();

        assert_eq!(plan.cacheable_prefix_len(), 0);
        assert_eq!(report.first_volatile_index, Some(0));
        assert_eq!(plan.stable_prefix_end, None);
        assert_eq!(plan.stable_prefix_hash(), None);
        assert_cache_metadata(&plan, &[false], &[None]);
        assert_token_report_invariants(&plan);
    }

    #[test]
    fn prompt_plan_cache_metadata_for_all_stable_segments() {
        let plan = build_cache_test_plan(
            &[PromptMessage::system("system")],
            &[HistoryItem::assistant("older assistant")],
            0,
        );
        let report = plan.token_report();

        assert_eq!(plan.cacheable_prefix_len(), 2);
        assert_eq!(report.first_volatile_index, None);
        assert_eq!(plan.stable_prefix_end, Some(1));
        assert!(plan.stable_prefix_hash().is_some());
        assert_cache_metadata(
            &plan,
            &[true, true],
            &[None, Some(PromptCacheBoundaryKind::StablePrefixEnd)],
        );
        assert_token_report_invariants(&plan);
    }

    #[test]
    fn prompt_plan_cache_metadata_stops_at_first_volatile_segment() {
        let plan = build_cache_test_plan(
            &[
                PromptMessage::system("system"),
                PromptMessage::developer_with_origin("clock", PromptMessageOrigin::RuntimeClock),
            ],
            &[HistoryItem::assistant("older assistant")],
            0,
        );
        let report = plan.token_report();

        assert_eq!(plan.cacheable_prefix_len(), 1);
        assert_eq!(report.first_volatile_index, Some(1));
        assert_eq!(plan.stable_prefix_end, Some(0));
        assert!(plan.stable_prefix_hash().is_some());
        assert_cache_metadata(
            &plan,
            &[true, false, false],
            &[
                Some(PromptCacheBoundaryKind::StablePrefixEnd),
                Some(PromptCacheBoundaryKind::VolatileRegionStart),
                None,
            ],
        );
        assert_token_report_invariants(&plan);
        assert!(report.stable_after_boundary_tokens > 0);
    }

    #[test]
    fn prompt_plan_context_summary_text_and_hash_use_rendered_material() {
        let summary = "目标\n- 修复 compaction";
        let rendered = render_context_summary(summary);
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[HistoryItem::context_summary(summary)]),
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        });

        assert_eq!(plan.stable_prefix_end, Some(0));
        assert_eq!(plan.segments[0].text, rendered);
        assert_eq!(
            plan.segments[0].content,
            PromptSegmentContent::Text {
                text: render_context_summary(summary)
            }
        );
        assert_eq!(
            plan.stable_prefix_hash(),
            Some(stable_hash_input(&format!("{}:{}", plan.segments[0].id, rendered)).as_str())
        );
    }

    #[test]
    fn prompt_plan_classifies_context_and_current_turn_contributors() {
        let history = vec![
            HistoryItem::ContextSummary {
                text: "[Context: Index]\n- src/main.rs".into(),
            },
            HistoryItem::AssistantToolCalls {
                text: Some("running tool".into()),
                calls: vec![HistoryToolCall {
                    call_id: "call-1".into(),
                    name: "read".into(),
                    arguments_json: "{}".into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: "{}".into(),
            },
            HistoryItem::UserMessage {
                content: UserMessageContent::new("follow up", Vec::new()),
            },
        ];
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&history),
            protected_suffix_len: 3,
            evidence_message: None,
            selected_evidence_ids: &[],
        });

        assert!(
            plan.contributors
                .iter()
                .any(|contributor| contributor.kind == PromptContributorKind::ContextIndex)
        );
        assert!(
            plan.contributors
                .iter()
                .any(|contributor| contributor.kind == PromptContributorKind::CurrentTurn)
        );
        assert!(
            plan.segments
                .iter()
                .any(|segment| segment.protection.protocol_boundary)
        );
    }

    #[test]
    fn history_classification_prioritizes_runtime_provenance_over_context_labels() {
        let frame = ProtocolFrame {
            runtime_frame_id: None,
            source_provenance: None,
            history_index: 0,
            item: ProtocolFrameItem::ContextSummary {
                text: "[Context: Index]\n- runtime material".into(),
            },
        };

        for source in [
            RuntimeSource::ContextView,
            RuntimeSource::ContextTree,
            RuntimeSource::FoldedOutput,
            RuntimeSource::SummaryArtifact,
            RuntimeSource::SessionState,
        ] {
            let classification =
                classify_history_frame(&frame, false, Some(&RuntimeFrameProvenance::new(source)));
            assert_eq!(classification.stability, PromptSegmentStability::Volatile);
            assert_eq!(classification.source, source);
        }

        let derived = classify_history_frame(
            &frame,
            false,
            Some(&RuntimeFrameProvenance::new(RuntimeSource::Derived)),
        );
        assert_eq!(derived.stability, PromptSegmentStability::Volatile);
        assert_eq!(derived.kind, PromptContributorKind::ContextIndex);

        let transcript = classify_history_frame(
            &frame,
            false,
            Some(&RuntimeFrameProvenance::new(RuntimeSource::Transcript)),
        );
        assert_eq!(transcript.stability, PromptSegmentStability::Stable);
        assert_eq!(transcript.kind, PromptContributorKind::TranscriptFrame);

        let skill_material = classify_history_frame(
            &frame,
            false,
            Some(&RuntimeFrameProvenance::new(
                RuntimeSource::PromptContributor,
            )),
        );
        assert_eq!(skill_material.stability, PromptSegmentStability::Stable);
        assert_eq!(skill_material.kind, PromptContributorKind::TranscriptFrame);
    }

    #[test]
    fn prompt_plan_ids_are_deterministic() {
        let input = PromptPlanBuildInput {
            protocol: ApiProtocol::Completions,
            model_id: "gpt-test",
            prelude: &[PromptMessage::system("sys")],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[HistoryItem::UserMessage {
                content: UserMessageContent::new("hello", Vec::new()),
            }]),
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        };

        let first = build_prompt_plan(input.clone());
        let second = build_prompt_plan(input);

        let first_ids = first
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect::<Vec<_>>();
        let second_ids = second
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect::<Vec<_>>();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn prompt_plan_inserts_evidence_before_last_user_segment() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[
                HistoryItem::assistant("older assistant"),
                HistoryItem::user("latest user"),
                HistoryItem::assistant("tail assistant"),
            ]),
            protected_suffix_len: 0,
            evidence_message: Some("Relevant evidence"),
            selected_evidence_ids: &["ev-1".to_string()],
        });

        let texts = plan
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            vec![
                "older assistant",
                "Relevant evidence",
                "latest user",
                "tail assistant"
            ]
        );
    }

    #[test]
    fn prompt_plan_classifies_internal_continuation_as_user() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[HistoryItem::internal_continuation(
                "continue",
            )]),
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        });

        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].role, PromptSegmentRole::User);
    }

    #[test]
    fn prompt_plan_multimodal_user_content_affects_text_key_and_hash() {
        let image = |data_url: &str| UserImageAttachment {
            id: "img-1".into(),
            label: "screen.png".into(),
            mime: "image/png".into(),
            data_url: data_url.into(),
        };
        let build = |data_url: &str| {
            build_prompt_plan(PromptPlanBuildInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                prelude: &[],
                snapshot: &RuntimeSnapshot::new("test"),
                selected_frames: &history_items_to_frames(&[HistoryItem::user_content(
                    UserMessageContent::new("", vec![image(data_url)]),
                )]),
                protected_suffix_len: 0,
                evidence_message: None,
                selected_evidence_ids: &[],
            })
        };

        let first = build("data:image/png;base64,AAAA");
        let second = build("data:image/png;base64,BBBB");

        assert!(first.segments[0].text.contains("[ImageAttachment id=img-1"));
        assert_ne!(first.segments[0].text, second.segments[0].text);
        assert_ne!(
            first.segments[0].source.source_key,
            second.segments[0].source.source_key
        );
        assert_ne!(first.segments[0].id, second.segments[0].id);
    }

    fn build_cache_test_plan(
        prelude: &[PromptMessage],
        history: &[HistoryItem],
        protected_suffix_len: usize,
    ) -> PromptPlan {
        build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude,
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(history),
            protected_suffix_len,
            evidence_message: None,
            selected_evidence_ids: &[],
        })
    }

    fn assert_cache_metadata(
        plan: &PromptPlan,
        expected_eligibility: &[bool],
        expected_boundaries: &[Option<PromptCacheBoundaryKind>],
    ) {
        assert_eq!(
            plan.segments
                .iter()
                .map(|segment| segment.cache.cache_eligible)
                .collect::<Vec<_>>(),
            expected_eligibility
        );
        assert_eq!(
            plan.segments
                .iter()
                .map(|segment| segment.cache.boundary)
                .collect::<Vec<_>>(),
            expected_boundaries
        );
    }

    fn assert_token_report_invariants(plan: &PromptPlan) {
        let report = plan.token_report();
        let finalized_total = plan
            .segments
            .iter()
            .map(|segment| segment.tokens.estimated_input_tokens.unwrap_or(0))
            .sum::<u64>();

        assert_eq!(
            report.stable_prompt_tokens + report.volatile_prompt_tokens,
            report.total_prompt_tokens
        );
        assert!(report.cacheable_prefix_tokens <= report.stable_prompt_tokens);
        assert_eq!(
            report.stable_after_boundary_tokens,
            report.stable_prompt_tokens - report.cacheable_prefix_tokens
        );
        assert_eq!(report.total_prompt_tokens, finalized_total);
        assert_eq!(plan.estimated_input_tokens(), finalized_total);
    }
}
