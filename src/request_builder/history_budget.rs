use anyhow::Result;

use super::{
    BudgetReport, EvidenceBudgetReport, HistoryItem, ModelRequestMetadata, PromptMessage,
    ProtocolFrame, ToolSpec, effective_input_budget_tokens_for_tool_tokens,
    estimate_history_item_tokens, estimate_prelude_tokens, estimate_tools_tokens,
    history_items_from_frames, validate_history_items_complete,
};

pub(super) fn ensure_protected_context_within_budget(
    input_budget: u64,
    prelude_tokens: u64,
    protected_tokens: u64,
    evidence_tokens: u64,
) -> Result<()> {
    let fixed_tokens = prelude_tokens
        .saturating_add(protected_tokens)
        .saturating_add(evidence_tokens);
    if fixed_tokens > input_budget {
        anyhow::bail!(
            "protected current context exceeds input budget: protected/current context tokens ({fixed_tokens}) exceed budget ({input_budget}); prelude={prelude_tokens}, protected={protected_tokens}, evidence={evidence_tokens}"
        );
    }
    Ok(())
}

pub(super) fn validate_model_metadata(model: ModelRequestMetadata) -> Result<()> {
    if let Some(effective_input_limit_tokens) = model.effective_input_limit_tokens {
        if effective_input_limit_tokens == 0 {
            anyhow::bail!("model.effective_input_limit_tokens must be greater than 0");
        }
    }
    if let Some(max_output_tokens) = model.max_output_tokens {
        if max_output_tokens > u32::MAX as u64 {
            anyhow::bail!("model.max_output_tokens must be at most {}", u32::MAX);
        }
    }
    if let Some(temperature) = model.temperature {
        validate_f32_range("model.temperature", temperature, 0.0, 2.0)?;
    }
    if let Some(top_p) = model.top_p {
        validate_f32_range("model.top_p", top_p, 0.0, 1.0)?;
    }
    Ok(())
}

fn validate_f32_range(label: &str, value: f32, min: f32, max: f32) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        anyhow::bail!("{label} must be between {min} and {max}");
    }
    Ok(())
}

pub(super) fn retain_history(
    prelude: &[PromptMessage],
    history: &[ProtocolFrame],
    protected_start_index: usize,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
    evidence_budget: EvidenceBudgetReport,
    required_fallback_tokens: u64,
) -> (Vec<ProtocolFrame>, BudgetReport) {
    let history_len = history.len();
    let protected_start = protected_start_index.min(history_len);
    let (older, protected) = history.split_at(protected_start);

    let prelude_tokens = estimate_prelude_tokens(prelude);
    let protected_tokens = estimate_protocol_frame_tokens(protected);
    let context_window = model.context_window_tokens();
    let tools_tokens = if model.supports_tools {
        estimate_tools_tokens(tools)
    } else {
        0
    };
    let input_budget = effective_input_budget_tokens_for_tool_tokens(model, tools_tokens);

    let mut retained_older = Vec::new();
    let mut retained_older_tokens = 0_u64;

    let fixed_tokens = prelude_tokens
        .saturating_add(protected_tokens)
        .saturating_add(evidence_budget.estimated_evidence_tokens)
        .saturating_add(required_fallback_tokens);

    if fixed_tokens < input_budget {
        for unit in retention_units(older).into_iter().rev() {
            let cost = estimate_protocol_frame_tokens(unit);
            let next = fixed_tokens
                .saturating_add(retained_older_tokens)
                .saturating_add(cost);
            if next > input_budget {
                break;
            }
            retained_older.extend(unit.iter().cloned().rev());
            retained_older_tokens = retained_older_tokens.saturating_add(cost);
        }
        retained_older.reverse();
    }

    let mut retained = Vec::with_capacity(retained_older.len() + protected.len());
    retained.extend(retained_older.iter().cloned());
    retained.extend(protected.iter().cloned());
    let retained_history_items = retained.len();
    let dropped_history_items = history_len.saturating_sub(retained_history_items);
    let retained_tokens = estimate_protocol_frame_tokens(&retained);
    let estimated_request_tokens = prelude_tokens
        .saturating_add(evidence_budget.estimated_evidence_tokens)
        .saturating_add(required_fallback_tokens)
        .saturating_add(retained_tokens)
        .saturating_add(tools_tokens);

    (
        retained,
        BudgetReport {
            context_window_tokens: context_window,
            input_budget_tokens: input_budget,
            estimated_request_tokens,
            estimated_prelude_tokens: prelude_tokens,
            estimated_protected_tokens: protected_tokens,
            protected_safe_ceiling_tokens: 0,
            protected_reserve_tokens: 0,
            estimated_foldable_protected_tokens: 0,
            estimated_provider_folded_protected_tokens: 0,
            estimated_unaddressable_protected_tokens: 0,
            provider_folded_output_count: 0,
            estimated_retained_history_tokens: retained_tokens,
            estimated_tools_tokens: tools_tokens,
            estimated_evidence_tokens: evidence_budget.estimated_evidence_tokens,
            estimated_required_fallback_tokens: required_fallback_tokens,
            original_history_items: history_len,
            retained_history_items,
            dropped_history_items,
            selected_evidence_items: evidence_budget.selected_evidence_items,
            dropped_evidence_items: evidence_budget.dropped_evidence_items,
            truncated: dropped_history_items > 0,
            plan_total_prompt_tokens: 0,
            plan_stable_prompt_tokens: 0,
            plan_volatile_prompt_tokens: 0,
            plan_cacheable_prefix_tokens: 0,
            plan_stable_after_boundary_tokens: 0,
        },
    )
}

/// A tool-call batch and all of its outputs are retained atomically.
fn retention_units(frames: &[ProtocolFrame]) -> Vec<&[ProtocolFrame]> {
    let transcript = validate_history_items_complete(&history_items_from_frames(frames), None)
        .expect("history was validated before retention");
    let mut group_end_by_start = std::collections::BTreeMap::new();
    for group in transcript.tool_call_groups {
        let end = group
            .tool_output_indexes
            .iter()
            .copied()
            .max()
            .unwrap_or(group.assistant_index);
        group_end_by_start.insert(group.assistant_index, end);
    }
    let mut units = Vec::new();
    let mut index = 0;
    while index < frames.len() {
        let end = group_end_by_start.get(&index).copied().unwrap_or(index);
        units.push(&frames[index..=end]);
        index = end + 1;
    }
    units
}

pub(super) fn expand_protected_start_to_group(
    history: &[HistoryItem],
    protected_start: usize,
) -> Result<usize> {
    let transcript = validate_history_items_complete(history, Some(protected_start))?;
    Ok(transcript
        .tool_call_groups
        .iter()
        .fold(protected_start, |start, group| {
            let group_end = group
                .tool_output_indexes
                .iter()
                .copied()
                .max()
                .unwrap_or(group.assistant_index);
            if group.assistant_index < start && group_end >= start {
                group.assistant_index
            } else {
                start
            }
        }))
}

fn estimate_protocol_frame_tokens(frames: &[ProtocolFrame]) -> u64 {
    frames
        .iter()
        .map(|frame| estimate_history_item_tokens(&frame.to_history_item()))
        .sum()
}

pub(super) fn current_user_query(history: &[HistoryItem], protected_start_index: usize) -> String {
    history
        .iter()
        .skip(protected_start_index.min(history.len()))
        .rev()
        .find_map(|item| match item {
            HistoryItem::UserMessage { content } => Some(content.text.clone()),
            _ => None,
        })
        .or_else(|| {
            history.iter().rev().find_map(|item| match item {
                HistoryItem::UserMessage { content } => Some(content.text.clone()),
                _ => None,
            })
        })
        .unwrap_or_default()
}

pub(super) fn evidence_budget_tokens(context_window_tokens: u64) -> u64 {
    context_window_tokens
        .saturating_mul(15)
        .saturating_div(100)
        .clamp(512, 3_000)
}
