use super::*;
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewOperation,
};
use crate::protocol_frames::{history_items_from_frames, history_items_to_frames};
use crate::request_builder::HistoryToolCall;
use crate::runtime_context::{RuntimeFrameProvenance, RuntimeSnapshot, SourceSpan};
use crate::user_content::{UserImageAttachment, UserMessageContent};

#[test]
fn prompt_plan_marks_stable_prefix_boundary() {
    let plan = build_prompt_plan(PromptPlanBuildInput {
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
        evidence_message: Some("Relevant evidence:\n- [ev-1] file_excerpt src/main.rs — summary"),
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
fn assistant_turn_without_calls_keeps_reasoning_and_replay_in_prompt_content() {
    let replay = crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
        r#"[{"type":"thinking","signature":"signed"}]"#,
    )
    .expect("replay state");
    let item = HistoryItem::AssistantTurn {
        text: Some("answer".into()),
        reasoning_content: Some("reasoning".into()),
        replay: Some(replay),
        calls: Vec::new(),
    };

    assert!(matches!(
        history_item_content(&item),
        PromptSegmentContent::AssistantToolCalls {
            text: Some(text),
            reasoning_content: Some(reasoning),
            replay: Some(replay),
            calls,
        } if text == "answer"
            && reasoning == "reasoning"
            && replay.payload.to_string().contains("signed")
            && calls.is_empty()
    ));
}

#[test]
fn prompt_composition_groups_request_material_by_context_category() {
    let plan = build_prompt_plan(PromptPlanBuildInput {
        model_id: "gpt-test",
        prelude: &[
            PromptMessage::system("system"),
            PromptMessage::developer_with_origin("skill body", PromptMessageOrigin::SkillMaterial),
        ],
        snapshot: &RuntimeSnapshot::new("test"),
        selected_frames: &history_items_to_frames(&[
            HistoryItem::user("question"),
            HistoryItem::assistant("answer"),
            HistoryItem::AssistantTurn {
                text: None,
                reasoning_content: None,
                replay: None,
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
