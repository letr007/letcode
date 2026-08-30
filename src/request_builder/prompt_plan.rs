#![allow(dead_code)]

use crate::config::ApiProtocol;
use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
use crate::request_builder::{
    BudgetReport, HistoryItem, HistoryToolCall, ModelRequestMetadata, PromptMessage,
    PromptMessageOrigin, PromptRole, ProtectedContextPolicy, ToolSpec,
    estimate_history_item_tokens,
};
use crate::runtime_context::{
    PromptContributorKind, RuntimeFrameId, RuntimeFrameProvenance, RuntimePromptRole,
    RuntimeSnapshot, RuntimeSource,
};
use crate::user_content::{UserImageAttachment, UserMessageContent};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static PLAN_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_plan_call_count() {
    PLAN_CALL_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn plan_call_count() -> usize {
    PLAN_CALL_COUNT.with(Cell::get)
}

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
        reasoning_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_wire: Option<String>,
        calls: Vec<HistoryToolCall>,
    },
    ToolOutput {
        call_id: String,
        output_json: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<UserImageAttachment>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptPlan {
    pub protocol: ApiProtocol,
    pub model_id: String,
    pub contributors: Vec<PromptContributor>,
    pub segments: Vec<PromptSegment>,
    pub stable_prefix_end: Option<usize>,
    /// Internal canonical cache boundaries, excluded from transcript and
    /// provider schemas.
    pub kernel_end_exclusive: usize,
    pub envelope_end_exclusive: usize,
}

/// Pure prompt selection boundary. It derives all provider-visible material
/// from an immutable runtime snapshot and never mutates session state.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PromptPlanner;

#[derive(Debug, Clone)]
pub(crate) struct PromptPlannerInput<'a> {
    pub protocol: ApiProtocol,
    pub model: ModelRequestMetadata,
    pub model_id: &'a str,
    pub prelude: &'a [PromptMessage],
    pub snapshot: &'a RuntimeSnapshot,
    pub tools: &'a [ToolSpec],
    pub frozen_evidence: Option<&'a super::FrozenEvidence>,
    pub protected_context_policy: ProtectedContextPolicy,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedPrompt {
    pub prompt_plan: PromptPlan,
    pub budget: BudgetReport,
    pub selected_evidence_ids: Vec<String>,
    pub selected_evidence_message: Option<String>,
}

impl PromptPlanner {
    pub(crate) fn plan(input: PromptPlannerInput<'_>) -> anyhow::Result<PlannedPrompt> {
        #[cfg(test)]
        PLAN_CALL_COUNT.with(|count| count.set(count.get().saturating_add(1)));
        input.snapshot.validate_references()?;
        let active_history_frames = super::provider_visible_protocol_frames(input.snapshot);
        let active_protected_start_index =
            super::protected_start_index_for_snapshot(input.snapshot, &active_history_frames);
        let effective = effective_runtime_prompt(
            input.prelude,
            &active_history_frames,
            active_protected_start_index,
        )?;
        let effective_prelude = effective.prelude;
        let effective_history_frames = effective.history_frames;
        let effective_history = super::history_items_from_frames(&effective_history_frames);
        let effective_protected_start_index = effective.protected_start_index;
        super::validate_history_items_complete(
            &effective_history,
            Some(effective_protected_start_index),
        )?;
        let effective_protected_start_index = super::expand_protected_start_to_group(
            &effective_history,
            effective_protected_start_index,
        )?;

        super::validate_model_metadata(input.model.clone())?;
        let context_window = input.model.context_window_tokens();
        let tools_tokens = if input.model.supports_tools {
            super::estimate_tools_tokens(input.tools)
        } else {
            0
        };
        let input_budget =
            super::effective_input_budget_tokens_for_tool_tokens(input.model.clone(), tools_tokens);
        let protected_start = effective_protected_start_index.min(effective_history.len());
        let protected_tokens =
            super::estimate_history_tokens(&effective_history[protected_start..]);
        let prelude_tokens = super::estimate_prelude_tokens(&effective_prelude);
        let evidence_budget = super::evidence_budget_tokens(context_window)
            .min(input_budget.saturating_sub(protected_tokens.saturating_add(prelude_tokens)));
        let current_query =
            super::current_user_query(&effective_history, effective_protected_start_index);
        let frozen = input.frozen_evidence;
        let (mut evidence_message, mut selected_evidence_ids, mut dropped_evidence_items) =
            if let Some(frozen) = frozen {
                let selected_current_evidence = input
                    .snapshot
                    .evidence
                    .iter()
                    .filter(|evidence| frozen.selected_ids.contains(&evidence.id))
                    .count();
                (
                    frozen.message.clone(),
                    frozen.selected_ids.clone(),
                    input
                        .snapshot
                        .evidence
                        .len()
                        .saturating_sub(selected_current_evidence),
                )
            } else if evidence_budget > 0 {
                crate::evidence::evidence_context_message(
                    &input.snapshot.evidence,
                    &current_query,
                    evidence_budget,
                )
            } else {
                (None, Vec::new(), input.snapshot.evidence.len())
            };
        let mut estimated_evidence_tokens = evidence_message
            .as_deref()
            .map(crate::evidence::estimate_evidence_tokens)
            .unwrap_or(0);
        let (frames, budget, protected_ceiling) = loop {
            let selected_evidence_items = if frozen.is_some() {
                input
                    .snapshot
                    .evidence
                    .iter()
                    .filter(|evidence| selected_evidence_ids.contains(&evidence.id))
                    .count()
            } else {
                selected_evidence_ids.len()
            };
            match select_history_with_required_fallbacks(
                input.snapshot,
                &effective_prelude,
                &effective_history_frames,
                effective_protected_start_index,
                protected_tokens,
                input.model.clone(),
                input.tools,
                estimated_evidence_tokens,
                selected_evidence_items,
                dropped_evidence_items,
                input.protected_context_policy,
            ) {
                Ok(selection) => break selection,
                Err(_) if frozen.is_none() && evidence_message.is_some() => {
                    evidence_message = None;
                    selected_evidence_ids.clear();
                    dropped_evidence_items = input.snapshot.evidence.len();
                    estimated_evidence_tokens = 0;
                }
                Err(error) => return Err(error),
            }
        };
        super::validate_history_items_complete(
            &super::history_items_from_frames(&frames),
            Some(effective_protected_start_index),
        )?;
        let prompt_plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: input.protocol,
            model_id: input.model_id,
            prelude: &effective_prelude,
            snapshot: input.snapshot,
            selected_frames: &frames,
            segment_order_offset: 0,
            protected_suffix_len: effective_history
                .len()
                .saturating_sub(effective_protected_start_index.min(effective_history.len())),
            evidence_message: evidence_message.as_deref(),
            selected_evidence_ids: &selected_evidence_ids,
        });
        let mut budget = budget;
        budget.estimated_protected_tokens = protected_tokens;
        budget.protected_safe_ceiling_tokens = protected_ceiling;
        budget.protected_reserve_tokens = input.protected_context_policy.reserve_tokens;
        budget.estimated_unaddressable_protected_tokens = protected_tokens;
        Ok(PlannedPrompt {
            prompt_plan,
            budget,
            selected_evidence_ids,
            selected_evidence_message: evidence_message,
        })
    }
}

struct EffectiveRuntimePrompt {
    prelude: Vec<PromptMessage>,
    history_frames: Vec<ProtocolFrame>,
    protected_start_index: usize,
}

fn select_history_with_required_fallbacks(
    snapshot: &RuntimeSnapshot,
    prelude: &[PromptMessage],
    history_frames: &[ProtocolFrame],
    protected_start_index: usize,
    protected_tokens: u64,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
    estimated_evidence_tokens: u64,
    selected_evidence_items: usize,
    dropped_evidence_items: usize,
    protected_context_policy: ProtectedContextPolicy,
) -> anyhow::Result<(Vec<ProtocolFrame>, BudgetReport, u64)> {
    let contributors = snapshot.active_prompt_payload_contributors();
    let mut fallback_tokens = 0u64;
    let mut frames = Vec::new();
    let mut budget = None;
    for _ in 0..=contributors.len() {
        let (selected, selected_budget) = super::retain_history(
            prelude,
            history_frames,
            protected_start_index,
            protected_tokens,
            model.clone(),
            tools,
            super::EvidenceBudgetReport {
                estimated_evidence_tokens,
                selected_evidence_items,
                dropped_evidence_items,
            },
            fallback_tokens,
        );
        let selected_ids = selected
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<std::collections::BTreeSet<_>>();
        let next = contributors
            .iter()
            .filter(|(contributor, _)| {
                !contributor
                    .source_frame_ids
                    .iter()
                    .any(|id| selected_ids.contains(id))
            })
            .map(|(_, frame)| {
                frame
                    .prompt_payload
                    .as_ref()
                    .map(|payload| {
                        estimate_history_item_tokens(&HistoryItem::ContextSummary {
                            text: payload.text.clone(),
                        })
                    })
                    .unwrap_or(0)
            })
            .sum::<u64>();
        frames = selected;
        budget = Some(selected_budget);
        if next == fallback_tokens {
            break;
        }
        fallback_tokens = next;
    }
    let budget = budget.expect("fixed-point selection executes at least once");
    if budget.estimated_required_fallback_tokens != fallback_tokens {
        anyhow::bail!("skill prompt fallback selection did not converge");
    }

    // Keep the same admission surface as before extraction: reserve required
    // detached payload tokens first, then optionally keep a protected reserve.
    let hard_protected_ceiling = budget
        .input_budget_tokens
        .saturating_sub(budget.estimated_prelude_tokens)
        .saturating_sub(budget.estimated_evidence_tokens)
        .saturating_sub(budget.estimated_required_fallback_tokens);
    let protected_ceiling = if protected_context_policy.enabled() {
        hard_protected_ceiling.saturating_sub(protected_context_policy.reserve_tokens)
    } else {
        hard_protected_ceiling
    };
    super::ensure_protected_context_within_budget(
        budget.input_budget_tokens,
        budget
            .estimated_prelude_tokens
            .saturating_add(budget.estimated_required_fallback_tokens),
        budget.estimated_protected_tokens,
        budget.estimated_evidence_tokens,
    )?;
    Ok((frames, budget, protected_ceiling))
}

/// Materializes provider-visible runtime material in canonical order.
fn effective_runtime_prompt(
    input_prelude: &[PromptMessage],
    active_history_frames: &[ProtocolFrame],
    active_protected_start_index: usize,
) -> anyhow::Result<EffectiveRuntimePrompt> {
    {
        let mut stable_prelude = input_prelude
            .iter()
            .filter(|message| {
                matches!(
                    message.origin,
                    PromptMessageOrigin::StaticPrelude
                        | PromptMessageOrigin::SkillCatalog
                        | PromptMessageOrigin::SkillMaterial
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        // Do not encode volatile developer packets as synthetic history
        // frames: that would make the protocol sequence non-canonical.
        stable_prelude.extend(
            input_prelude
                .iter()
                .filter(|message| {
                    !matches!(
                        message.origin,
                        PromptMessageOrigin::StaticPrelude
                            | PromptMessageOrigin::SkillCatalog
                            | PromptMessageOrigin::SkillMaterial
                    )
                })
                .cloned(),
        );
        let active_protected_start_index = super::expand_protected_start_to_group(
            &super::history_items_from_frames(active_history_frames),
            active_protected_start_index,
        )?
        .min(active_history_frames.len());
        let protected_start_index = active_protected_start_index;
        let history_frames = active_history_frames.to_vec();
        Ok(EffectiveRuntimePrompt {
            prelude: stable_prelude,
            history_frames,
            protected_start_index,
        })
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCompositionEntry {
    #[serde(default, alias = "key")]
    pub category: String,
    #[serde(default, alias = "tokens")]
    pub estimated_tokens: u64,
    #[serde(default)]
    pub segments: usize,
}

impl PromptPlan {
    pub(crate) fn recompute_cache_metadata(&mut self) {
        self.stable_prefix_end = None;
        for segment in &mut self.segments {
            segment.cache = PromptCacheMetadata {
                cache_eligible: false,
                boundary: None,
                prefix_hash: None,
            };
        }
        let stable_prefix_len = self.cacheable_prefix_len();
        if let Some(stable_end) = stable_prefix_len.checked_sub(1) {
            let prefix_hash = stable_hash_input(
                &self.segments[..=stable_end]
                    .iter()
                    .map(|segment| format!("{}:{}", segment.id, segment.text))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            self.stable_prefix_end = Some(stable_end);
            self.segments[stable_end].cache.boundary =
                Some(PromptCacheBoundaryKind::StablePrefixEnd);
            self.segments[stable_end].cache.prefix_hash = Some(prefix_hash.clone());
            if let Some(segment) = self.segments.get_mut(stable_end + 1) {
                segment.cache.boundary = Some(PromptCacheBoundaryKind::VolatileRegionStart);
                segment.cache.prefix_hash = Some(prefix_hash);
            }
        }
        for segment in self.segments.iter_mut().take(stable_prefix_len) {
            segment.cache.cache_eligible = true;
        }
    }
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

    pub(crate) fn composition(&self, tool_definition_tokens: u64) -> Vec<PromptCompositionEntry> {
        let mut entries = Vec::<PromptCompositionEntry>::new();
        for segment in &self.segments {
            let category = composition_category(segment.source.contributor_kind).to_string();
            let tokens = segment.tokens.estimated_input_tokens.unwrap_or(0);
            if let Some(entry) = entries.iter_mut().find(|entry| entry.category == category) {
                entry.estimated_tokens = entry.estimated_tokens.saturating_add(tokens);
                entry.segments = entry.segments.saturating_add(1);
            } else {
                entries.push(PromptCompositionEntry {
                    category,
                    estimated_tokens: tokens,
                    segments: 1,
                });
            }
        }
        if tool_definition_tokens > 0 {
            entries.push(PromptCompositionEntry {
                category: "tools".into(),
                estimated_tokens: tool_definition_tokens,
                segments: 1,
            });
        }
        entries
    }

    pub(crate) fn stable_prefix_hash(&self) -> Option<&str> {
        self.segments.iter().find_map(|segment| {
            (segment.cache.boundary == Some(PromptCacheBoundaryKind::StablePrefixEnd))
                .then_some(segment.cache.prefix_hash.as_deref())
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
    pub segment_order_offset: usize,
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
    build_prompt_plan_with_runtime_material(input, true)
}

pub(crate) fn build_prompt_plan_suffix(
    protocol: ApiProtocol,
    model_id: &str,
    snapshot: &RuntimeSnapshot,
    selected_frames: &[ProtocolFrame],
    segment_order_offset: usize,
) -> PromptPlan {
    let mut suffix = build_prompt_plan_with_runtime_material(
        PromptPlanBuildInput {
            protocol,
            model_id,
            prelude: &[],
            snapshot,
            selected_frames,
            segment_order_offset,
            protected_suffix_len: selected_frames.len(),
            evidence_message: None,
            selected_evidence_ids: &[],
        },
        false,
    );
    canonicalize_prompt_plan_preserving_offset(&mut suffix, segment_order_offset);
    suffix
}

fn build_prompt_plan_with_runtime_material(
    input: PromptPlanBuildInput<'_>,
    include_runtime_material: bool,
) -> PromptPlan {
    let mut builder =
        PromptPlanBuilder::new(input.protocol, input.model_id, input.segment_order_offset);

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
    let prompt_material = if include_runtime_material {
        input
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
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
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
            classify_history_frame(frame, index >= current_turn_start, provenance.as_ref());
        if !matches!(
            frame.item,
            ProtocolFrameItem::AssistantToolCalls { .. } | ProtocolFrameItem::ToolOutput { .. }
        ) && let Some(contributor) = frame.runtime_frame_id.and_then(|id| {
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

fn canonicalize_prompt_plan_preserving_offset(plan: &mut PromptPlan, order_offset: usize) {
    let mut history = Vec::new();
    let mut current = Vec::new();
    for segment in plan.segments.drain(..) {
        if segment.protection.current_turn
            || segment.source.contributor_kind == PromptContributorKind::CurrentTurn
        {
            current.push(segment);
        } else {
            history.push(segment);
        }
    }
    history.append(&mut current);
    for (index, segment) in history.iter_mut().enumerate() {
        let order = order_offset.saturating_add(index) as u32;
        segment.order = order;
        segment.source.order = order;
        segment.stability = PromptSegmentStability::Volatile;
    }
    plan.segments = history;
    plan.kernel_end_exclusive = 0;
    plan.envelope_end_exclusive = 0;
    for contributor in &mut plan.contributors {
        contributor.segment_ids.clear();
    }
    for segment in &plan.segments {
        if let Some(contributor) = plan
            .contributors
            .iter_mut()
            .find(|contributor| contributor.id == segment.contributor_id)
        {
            contributor.segment_ids.push(segment.id.clone());
        }
    }
    plan.contributors
        .retain(|contributor| !contributor.segment_ids.is_empty());
    for (index, contributor) in plan.contributors.iter_mut().enumerate() {
        contributor.order = order_offset.saturating_add(index) as u32;
    }
    plan.recompute_cache_metadata();
}

/// Normalize the sole canonical prompt projection after selection.
pub(crate) fn canonicalize_prompt_plan(mut plan: PromptPlan) -> PromptPlan {
    let mut kernel = Vec::new();
    let mut envelope = Vec::new();
    let mut evidence = Vec::new();
    let mut durable = Vec::new();
    let mut history = Vec::new();
    let mut current = Vec::new();
    for segment in plan.segments.drain(..) {
        // Protocol frames take precedence over contributor classification. In
        // particular, a resolved skill output must remain beside the
        // assistant call that introduced it rather than moving into kernel
        // material ahead of that call.
        if matches!(
            segment.content,
            PromptSegmentContent::AssistantToolCalls { .. }
                | PromptSegmentContent::ToolOutput { .. }
        ) {
            if segment.protection.current_turn {
                current.push(segment);
            } else {
                history.push(segment);
            }
        } else if matches!(
            segment.source.contributor_kind,
            PromptContributorKind::SystemPrelude
                | PromptContributorKind::DeveloperPrelude
                | PromptContributorKind::SkillMaterial
        ) && segment.stability == PromptSegmentStability::Stable
        {
            kernel.push(segment);
        } else if segment.source.source_label.as_deref() == Some("prelude") {
            envelope.push(segment);
        } else if segment.source.contributor_kind == PromptContributorKind::Evidence {
            evidence.push(segment);
        } else if matches!(
            segment.source.contributor_kind,
            PromptContributorKind::ContextMaterial | PromptContributorKind::ContextIndex
        ) {
            durable.push(segment);
        } else if segment.protection.current_turn
            || segment.source.contributor_kind == PromptContributorKind::CurrentTurn
        {
            current.push(segment);
        } else {
            history.push(segment);
        }
    }
    let kernel_end_exclusive = kernel.len();
    // The envelope contains both prelude envelope material and leading
    // evidence.  Both boundaries are exclusive segment counts.
    let envelope_end_exclusive = kernel.len() + envelope.len() + evidence.len();
    kernel.append(&mut envelope);
    kernel.append(&mut evidence);
    kernel.append(&mut durable);
    kernel.append(&mut history);
    kernel.append(&mut current);
    for (order, segment) in kernel.iter_mut().enumerate() {
        segment.order = order as u32;
        segment.source.order = order as u32;
        // A cold canonical request caches only its kernel. Subsequent epoch previews
        // promote committed segments explicitly.
        segment.stability = if order < kernel_end_exclusive {
            PromptSegmentStability::Stable
        } else {
            PromptSegmentStability::Volatile
        };
    }
    plan.segments = kernel;
    plan.kernel_end_exclusive = kernel_end_exclusive;
    plan.envelope_end_exclusive = envelope_end_exclusive;
    plan.recompute_cache_metadata();
    plan
}

fn last_user_frame_index(frames: &[ProtocolFrame]) -> Option<usize> {
    frames
        .iter()
        .rposition(|frame| matches!(frame.item, ProtocolFrameItem::UserMessage { .. }))
}

struct PromptPlanBuilder {
    protocol: ApiProtocol,
    model_id: String,
    segment_order_offset: usize,
    contributors: Vec<PromptContributor>,
    segments: Vec<PromptSegment>,
}

impl PromptPlanBuilder {
    fn new(protocol: ApiProtocol, model_id: &str, segment_order_offset: usize) -> Self {
        Self {
            protocol,
            model_id: model_id.to_string(),
            segment_order_offset,
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
                order: self
                    .segment_order_offset
                    .saturating_add(self.contributors.len()) as u32,
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
        let order = self
            .segment_order_offset
            .saturating_add(self.segments.len()) as u32;
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
            kernel_end_exclusive: 0,
            envelope_end_exclusive: 0,
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

fn composition_category(kind: PromptContributorKind) -> &'static str {
    match kind {
        PromptContributorKind::SystemPrelude | PromptContributorKind::DeveloperPrelude => "system",
        PromptContributorKind::SkillMaterial => "skills",
        PromptContributorKind::RuntimeContext
        | PromptContributorKind::ContextMaterial
        | PromptContributorKind::ContextIndex
        | PromptContributorKind::Evidence => "context",
        PromptContributorKind::TranscriptFrame
        | PromptContributorKind::CurrentTurn
        | PromptContributorKind::Other => "messages",
    }
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
        PromptMessageOrigin::SkillMaterial => (
            PromptContributorKind::SkillMaterial,
            "skill_material",
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

    // Context indexes are durable provider-visible frames regardless of
    // whether they were materialized by the runtime view or restored from
    // history. Handle them before mutable-runtime classification.
    if let ProtocolFrameItem::ContextSummary { text } = &frame.item
        && text.starts_with("[Context: Index]")
    {
        return HistoryClassification {
            kind: PromptContributorKind::ContextIndex,
            label: Some("context_index".to_string()),
            role: PromptSegmentRole::Developer,
            stability: PromptSegmentStability::Volatile,
            retention: PromptSegmentRetention::Retained,
            source: provenance.map_or(RuntimeSource::ContextView, |provenance| provenance.source),
        };
    }

    if let Some(provenance) = provenance
        && is_mutable_runtime_projection(provenance.source)
    {
        return HistoryClassification {
            kind: runtime_projection_contributor_kind(provenance.source),
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
            | RuntimeSource::SummaryArtifact
            | RuntimeSource::SessionState
    )
}

fn runtime_projection_contributor_kind(source: RuntimeSource) -> PromptContributorKind {
    match source {
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
        HistoryItem::AssistantToolCalls { text, calls, .. } => {
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
            images,
        } => format!(
            "{call_id}\n{output_json}{}",
            images
                .iter()
                .map(UserImageAttachment::prompt_plan_placeholder)
                .collect::<Vec<_>>()
                .join("\n")
        ),
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
        HistoryItem::AssistantToolCalls {
            text,
            reasoning_content,
            reasoning_wire,
            calls,
        } => PromptSegmentContent::AssistantToolCalls {
            text: text.clone(),
            reasoning_content: reasoning_content.clone(),
            reasoning_wire: reasoning_wire.clone(),
            calls: calls.clone(),
        },
        HistoryItem::ToolOutput {
            call_id,
            output_json,
            images,
        } => PromptSegmentContent::ToolOutput {
            call_id: call_id.clone(),
            output_json: output_json.clone(),
            images: images.clone(),
        },
    }
}

fn estimate_prompt_message_tokens(message: &PromptMessage) -> Option<u64> {
    let json_len = serde_json::to_string(message).ok()?.len();
    Some((json_len as u64).div_ceil(3).saturating_add(8))
}

fn estimate_text_tokens(text: &str) -> u64 {
    (text.len() as u64).div_ceil(3).saturating_add(8)
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
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewOperation,
    };
    use crate::protocol_frames::{history_items_from_frames, history_items_to_frames};
    use crate::request_builder::HistoryToolCall;
    use crate::runtime_context::{RuntimeFrameProvenance, RuntimeSnapshot, SourceSpan};
    use crate::user_content::{UserImageAttachment, UserMessageContent};

    /// Produces an ASCII-only canonical ToolResult with an exact serialized
    /// byte count. The fixed framing is measured, never guessed.
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
            segment_order_offset: 0,
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
    fn prompt_composition_groups_request_material_by_context_category() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[
                PromptMessage::system("system"),
                PromptMessage::developer_with_origin(
                    "skill body",
                    PromptMessageOrigin::SkillMaterial,
                ),
            ],
            snapshot: &RuntimeSnapshot::new("test"),
            selected_frames: &history_items_to_frames(&[
                HistoryItem::user("question"),
                HistoryItem::assistant("answer"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    reasoning_content: None,
                    reasoning_wire: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-1".into(),
                        name: "fs__read".into(),
                        arguments_json: "{}".into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-1".into(),
                    output_json: "{}".into(),
                    images: Vec::new(),
                },
            ]),
            segment_order_offset: 0,
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        });

        let composition = plan.composition(321);
        assert!(
            composition
                .iter()
                .any(|entry| { entry.category == "system" && entry.estimated_tokens > 0 })
        );
        assert!(
            composition
                .iter()
                .any(|entry| { entry.category == "skills" && entry.estimated_tokens > 0 })
        );
        assert!(
            composition
                .iter()
                .any(|entry| { entry.category == "messages" && entry.estimated_tokens > 0 })
        );
        assert!(
            composition
                .iter()
                .any(|entry| { entry.category == "tools" && entry.estimated_tokens == 321 })
        );
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
            segment_order_offset: 0,
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
